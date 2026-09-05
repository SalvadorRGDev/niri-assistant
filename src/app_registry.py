"""
Whitelist de aplicaciones y rutinas abribles por voz.

Mismo principio de seguridad que ALLOWED_ROOTS en paths.py: el NLU nunca
decide qué comando ejecutar, solo un alias. `nombre` (lo que devuelve el
NLU) se busca en APP_ALIASES y, si no matchea ninguna clave, la acción se
rechaza — nunca se ejecuta un string crudo que venga del modelo.

Comandos verificados contra los .desktop reales del sistema (Hyprland,
CachyOS), no adivinados.
"""
from typing import Dict, List

APP_ALIASES: Dict[str, List[str]] = {
    "spotify": ["spotify-launcher"],
    "antigravity": ["/opt/antigravity-ide/antigravity-ide"],
    "brave": ["brave"],
    "navegador": ["brave"],
    "discord": ["/usr/bin/discord", "--url", "--"],
    "steam": ["/usr/bin/steam"],
    "firefox": ["/usr/lib/firefox/firefox"],
    "chromium": ["chromium"],
    "codigo": ["code-oss"],
    "vscode": ["code-oss"],
}

# Rutina -> lista de alias de APP_ALIASES a abrir en orden.
ROUTINES: Dict[str, List[str]] = {
    "modo_programador": ["antigravity", "spotify"],
}


def resolve_app(nombre: str) -> List[str]:
    """Devuelve el comando (lista de argv) para `nombre`, o [] si no está en la whitelist."""
    if not nombre:
        return []
    return APP_ALIASES.get(nombre.strip().lower(), [])


def resolve_routine(nombre: str) -> List[str]:
    """Devuelve la lista de alias de apps que componen la rutina `nombre`, o [] si no existe."""
    if not nombre:
        return []
    return ROUTINES.get(nombre.strip().lower(), [])
