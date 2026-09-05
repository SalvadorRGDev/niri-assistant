import os
import shutil
from pathlib import Path
from typing import List, Optional

from src.schemas import FileAction, ExecutionResult
from src.logger import get_logger
from src.audit import log_action
from src.paths import ALLOWED_ROOTS, DEFAULT_ROOT, speakable_path

logger = get_logger("Executor")

# Acciones que exigen confirmación explícita por voz antes de tocar disco
# (capa 5 del plan: "Confirmación por voz obligatoria en eliminar/mover").
CONFIRMATION_REQUIRED = {"eliminar", "mover"}

# Límite de caracteres leídos en voz alta para la acción "leer" (evita
# que Niri intente narrar un archivo entero).
LEER_MAX_CHARS = 800


class Executor:
    def is_safe_path(self, target_path: Path) -> bool:
        """Check if the resolved path is within the allowed roots."""
        resolved = target_path.resolve()
        for root in ALLOWED_ROOTS:
            try:
                # relative_to will throw ValueError if resolved is not inside root
                resolved.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    def is_protected_root(self, target_path: Path) -> bool:
        """True si target_path ES uno de los roots permitidos (no una subcarpeta).
        Nunca se permite borrar o mover un root completo, aunque técnicamente
        esté "dentro" de la whitelist."""
        resolved = target_path.resolve()
        return any(resolved == root for root in ALLOWED_ROOTS)

    def find_existing_roots(self, nombre: str) -> List[Path]:
        """
        Devuelve la lista de roots permitidos donde existe `nombre` (puede
        incluir subcarpetas, p.ej. 'trabajo/fotos'). Se usa para localizar
        un archivo/carpeta existente cuando el usuario no dijo en qué
        carpeta está (eliminar, mover, leer).
        """
        matches = []
        for root in ALLOWED_ROOTS:
            candidate = root / nombre
            if self.is_safe_path(candidate) and candidate.exists():
                matches.append(root)
        return matches

    def execute(
        self,
        action: FileAction,
        base_dir: Path = DEFAULT_ROOT,
        raw_text: Optional[str] = None,
        confirmed: bool = False,
        destino_dir: Optional[Path] = None,
    ) -> ExecutionResult:
        """
        Ejecuta (o, para eliminar/mover sin `confirmed=True`, solo valida y
        prepara la pregunta de confirmación) y audita el intento en el log
        append-only, sin importar el desenlace.

        `base_dir` es la carpeta raíz ya resuelta (por src/paths.py o por
        MainLoop tras preguntarle al usuario) donde vive/se crea `nombre`.
        `destino_dir`, si se da, es la carpeta de destino ya resuelta para
        'mover'; si no se da, se resuelve como `base_dir / action.destino`.
        """
        result = self._execute(action, base_dir=base_dir, confirmed=confirmed, destino_dir=destino_dir)
        log_action(
            raw_text,
            action.model_dump(),
            result.text,
            extra={
                "confirmado": confirmed,
                "requiere_confirmacion": result.needs_confirmation,
                "base_dir": str(base_dir),
            },
        )
        return result

    def _execute(self, action: FileAction, base_dir: Path, confirmed: bool, destino_dir: Optional[Path]) -> ExecutionResult:
        if action.action == "ninguna":
            logger.info("Executor: Ninguna acción a realizar.")
            return ExecutionResult(text="No entendí la acción, o no hay nada que deba hacer.")

        if action.action != "listar" and not action.nombre:
            logger.warning("Executor: Se requiere un nombre para la acción, pero no se proporcionó.")
            return ExecutionResult(text="Lo siento, necesito que me digas un nombre para el archivo o carpeta.")

        target_path = base_dir / action.nombre if action.nombre else base_dir

        if not self.is_safe_path(target_path):
            logger.error(f"Security Alert: Attempt to access path outside allowed roots: {target_path}")
            return ExecutionResult(text="Por seguridad, no tengo permitido acceder a esa ruta.")

        # --- Confirmación obligatoria para acciones destructivas/irreversibles ---
        if action.action in CONFIRMATION_REQUIRED and not confirmed:
            if self.is_protected_root(target_path):
                logger.error(f"Security Alert: Attempt to {action.action} a protected root: {target_path}")
                return ExecutionResult(text="Por seguridad, no puedo tocar esa carpeta raíz.")

            if action.action == "mover":
                dest_path = destino_dir if destino_dir else (base_dir / action.destino if action.destino else None)
                if dest_path is None:
                    return ExecutionResult(text="Necesito que me digas a dónde quieres mover eso.")
                if not self.is_safe_path(dest_path):
                    logger.error(f"Security Alert: Attempt to move into unsafe path: {dest_path}")
                    return ExecutionResult(text="Por seguridad, no tengo permitido mover nada a ese destino.")
                pregunta = (
                    f"Vas a mover '{action.nombre}' de {speakable_path(base_dir)} "
                    f"a {speakable_path(dest_path)}. Di 'sí' para confirmar, o 'no' para cancelar."
                )
            else:  # eliminar
                pregunta = (
                    f"Vas a eliminar '{action.nombre}' de {speakable_path(base_dir)} "
                    "de forma permanente. Di 'sí' para confirmar, o 'no' para cancelar."
                )
            logger.info(f"Pidiendo confirmación de voz para: {action.action} -> {target_path}")
            return ExecutionResult(text=pregunta, needs_confirmation=True)

        if action.action == "crear_carpeta":
            try:
                target_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Éxito: Carpeta creada en {target_path}")
                return ExecutionResult(text=f"He creado la carpeta {action.nombre} en {speakable_path(base_dir)}.")
            except Exception as e:
                logger.error(f"Error creando carpeta: {e}")
                return ExecutionResult(text="Hubo un error al intentar crear la carpeta.")

        elif action.action == "crear_archivo":
            try:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with open(target_path, 'w') as f:
                    if action.contenido:
                        f.write(action.contenido)
                logger.info(f"Éxito: Archivo creado en {target_path}")
                return ExecutionResult(text=f"Archivo {action.nombre} creado en {speakable_path(base_dir)}.")
            except Exception as e:
                logger.error(f"Error creando archivo: {e}")
                return ExecutionResult(text="No pude crear el archivo debido a un error.")

        elif action.action == "eliminar":
            # Solo se llega aquí con confirmed=True (ver bloque de confirmación arriba)
            if not target_path.exists():
                return ExecutionResult(text=f"No encontré '{action.nombre}' para eliminar.")
            try:
                if target_path.is_dir():
                    shutil.rmtree(target_path)
                else:
                    target_path.unlink()
                logger.warning(f"ELIMINADO (confirmado por voz): {target_path}")
                return ExecutionResult(text=f"Eliminé {action.nombre} de {speakable_path(base_dir)} correctamente.")
            except Exception as e:
                logger.error(f"Error eliminando: {e}")
                return ExecutionResult(text="Hubo un error al intentar eliminar.")

        elif action.action == "mover":
            # Solo se llega aquí con confirmed=True (ver bloque de confirmación arriba)
            dest_path = destino_dir if destino_dir else (base_dir / action.destino if action.destino else None)
            if dest_path is None:
                return ExecutionResult(text="Necesito que me digas a dónde quieres mover eso.")
            if not target_path.exists():
                return ExecutionResult(text=f"No encontré '{action.nombre}' para mover.")
            if not self.is_safe_path(dest_path):
                logger.error(f"Security Alert: Attempt to move into unsafe path: {dest_path}")
                return ExecutionResult(text="Por seguridad, no tengo permitido mover nada a ese destino.")

            try:
                # dest_path es siempre la CARPETA destino (nunca un rename):
                # "mover X a Y" -> X termina dentro de Y conservando su nombre.
                # Si Y no existe todavía, se crea. Esto evita que shutil.move
                # renombre X a Y cuando Y no exista de antemano.
                dest_path.mkdir(parents=True, exist_ok=True)
                final_path = dest_path / Path(action.nombre).name
                if final_path.exists():
                    return ExecutionResult(text=f"Ya existe algo llamado {action.nombre} en {speakable_path(dest_path)}, no lo muevo para no sobrescribirlo.")
                shutil.move(str(target_path), str(final_path))
                logger.warning(f"MOVIDO (confirmado por voz): {target_path} -> {final_path}")
                return ExecutionResult(text=f"Moví {action.nombre} a {speakable_path(dest_path)} correctamente.")
            except Exception as e:
                logger.error(f"Error moviendo: {e}")
                return ExecutionResult(text="Hubo un error al intentar mover el archivo.")

        elif action.action == "listar":
            try:
                items = os.listdir(base_dir)
                logger.info(f"Contenido de {base_dir}: {', '.join(items)}")
                return ExecutionResult(text=f"La carpeta {speakable_path(base_dir)} tiene {len(items)} elementos.")
            except Exception as e:
                logger.error(f"Error al listar: {e}")
                return ExecutionResult(text="Hubo un problema al leer la carpeta.")

        elif action.action == "leer":
            if not target_path.exists() or not target_path.is_file():
                return ExecutionResult(text=f"No encontré el archivo '{action.nombre}' para leer.")
            try:
                with open(target_path, 'r', encoding='utf-8', errors='ignore') as f:
                    contenido = f.read(LEER_MAX_CHARS + 1)
                if not contenido.strip():
                    return ExecutionResult(text=f"El archivo {action.nombre} está vacío.")
                truncado = len(contenido) > LEER_MAX_CHARS
                contenido = contenido[:LEER_MAX_CHARS]
                logger.info(f"Éxito: Leído {target_path} ({len(contenido)} caracteres, truncado={truncado})")
                sufijo = ", y sigue, pero ahí lo corto" if truncado else ""
                return ExecutionResult(text=f"El archivo {action.nombre} dice: {contenido}{sufijo}.")
            except Exception as e:
                logger.error(f"Error leyendo archivo: {e}")
                return ExecutionResult(text="Hubo un error al intentar leer el archivo.")

        else:
            logger.warning(f"Acción no implementada todavía: {action.action}")
            return ExecutionResult(text="Todavía no sé cómo hacer esa acción.")
