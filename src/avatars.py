"""Bespoke inline SVG cartoon avatars for AI-council-sim personas and Meta-Judges."""

from __future__ import annotations

PERSONA_ACCENT_COLOR: dict[str, str] = {
    "retail_fomo": "#D97706",
    "fundamentalist": "#16243D",
    "paranoid_bear": "#9B1C1C",
    "contrarian": "#475569",
    "macro_analyst": "#146C43",
    "judge_1": "#16243D",
    "judge_2": "#0F766E",
    "judge_3": "#9A5B13",
}


def judge_display_name(judge_id: str) -> str:
    """Convert internal judge IDs ('judge_1') into clean human-readable labels ('Judge 1')."""
    clean = judge_id.strip()
    if clean.lower().startswith("judge_"):
        suffix = clean.split("_", 1)[1]
        if suffix.isdigit():
            return f"Judge {suffix}"
    return clean.replace("_", " ").title() if clean else "Judge"


def _retail_fomo_svg(size: int) -> str:
    # Excited young momentum trader with orange headband, starry eyes & big grin
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<circle cx="24" cy="24" r="23" fill="#FEF3C7" stroke="#D97706" stroke-width="2"/>'
        '<path d="M12 42C12 35.5 17 32 24 32C31 32 36 35.5 36 42" fill="#D97706"/>'
        '<circle cx="24" cy="22" r="11" fill="#FDE2C4" stroke="#1F2937" stroke-width="1.6"/>'
        '<path d="M13 19C13.5 12 18 9 24 9C30 9 34.5 12 35 19C32 16.5 16 16.5 13 19Z" fill="#7C2D12"/>'
        '<rect x="13" y="14.5" width="22" height="3.6" rx="1.8" fill="#EF4444"/>'
        '<circle cx="19.5" cy="22.5" r="2.1" fill="#1F2937"/>'
        '<circle cx="28.5" cy="22.5" r="2.1" fill="#1F2937"/>'
        '<circle cx="20.2" cy="21.8" r="0.7" fill="#FFFFFF"/>'
        '<circle cx="29.2" cy="21.8" r="0.7" fill="#FFFFFF"/>'
        '<path d="M19.5 27.2C21 29.4 27 29.4 28.5 27.2" stroke="#1F2937" stroke-width="1.8" stroke-linecap="round"/>'
        '<path d="M35 10L37 6L39 10L37 9L35 10Z" fill="#D97706"/>'
        "</svg>"
    )


def _fundamentalist_svg(size: int) -> str:
    # Wise bespectacled senior value investor in navy suit & tie
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<circle cx="24" cy="24" r="23" fill="#E8EEF6" stroke="#16243D" stroke-width="2"/>'
        '<path d="M11 43C11.5 35.5 16.5 32 24 32C31.5 32 36.5 35.5 37 43" fill="#16243D"/>'
        '<path d="M22 33L24 38L26 33Z" fill="#FFFFFF"/>'
        '<path d="M23.2 35.5L24 41L24.8 35.5Z" fill="#9B1C1C"/>'
        '<circle cx="24" cy="22" r="10.8" fill="#F8D8B8" stroke="#16243D" stroke-width="1.6"/>'
        '<path d="M13.5 19.5C14 13 18 10.5 24 10.5C30 10.5 34 13 34.5 19.5C32 16.5 28.5 15 24 15C19.5 15 16 16.5 13.5 19.5Z" fill="#94A3B8"/>'
        '<circle cx="19.5" cy="22.5" r="3.4" fill="#FFFFFF" fill-opacity="0.5" stroke="#16243D" stroke-width="1.5"/>'
        '<circle cx="28.5" cy="22.5" r="3.4" fill="#FFFFFF" fill-opacity="0.5" stroke="#16243D" stroke-width="1.5"/>'
        '<line x1="22.9" y1="22.5" x2="25.1" y2="22.5" stroke="#16243D" stroke-width="1.5"/>'
        '<circle cx="19.5" cy="22.5" r="1.2" fill="#16243D"/>'
        '<circle cx="28.5" cy="22.5" r="1.2" fill="#16243D"/>'
        '<path d="M21 28C22.8 29.2 25.2 29.2 27 28" stroke="#16243D" stroke-width="1.6" stroke-linecap="round"/>'
        "</svg>"
    )


def _paranoid_bear_svg(size: int) -> str:
    # Skeptical grumpy cartoon bear with furrowed eyebrows & red tie
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<circle cx="24" cy="24" r="23" fill="#FCEEEE" stroke="#9B1C1C" stroke-width="2"/>'
        '<circle cx="14.5" cy="14" r="4.2" fill="#78350F" stroke="#1F2937" stroke-width="1.5"/>'
        '<circle cx="33.5" cy="14" r="4.2" fill="#78350F" stroke="#1F2937" stroke-width="1.5"/>'
        '<circle cx="14.5" cy="14" r="1.8" fill="#D97706"/>'
        '<circle cx="33.5" cy="14" r="1.8" fill="#D97706"/>'
        '<path d="M12 42C12.5 35.5 17 32.5 24 32.5C31 32.5 35.5 35.5 36 42" fill="#9B1C1C"/>'
        '<circle cx="24" cy="23" r="11" fill="#92400E" stroke="#1F2937" stroke-width="1.6"/>'
        '<ellipse cx="24" cy="26.2" rx="5.5" ry="4.2" fill="#FDE68A"/>'
        '<ellipse cx="24" cy="24.5" rx="2.1" ry="1.4" fill="#1F2937"/>'
        '<line x1="16.5" y1="19" x2="21.5" y2="20.6" stroke="#1F2937" stroke-width="2" stroke-linecap="round"/>'
        '<line x1="31.5" y1="19" x2="26.5" y2="20.6" stroke="#1F2937" stroke-width="2" stroke-linecap="round"/>'
        '<circle cx="19.3" cy="22" r="1.4" fill="#1F2937"/>'
        '<circle cx="28.7" cy="22" r="1.4" fill="#1F2937"/>'
        '<path d="M21 28.4C22.8 27.3 25.2 27.3 27 28.4" stroke="#1F2937" stroke-width="1.6" stroke-linecap="round"/>'
        "</svg>"
    )


def _contrarian_svg(size: int) -> str:
    # Sly independent thinker with raised eyebrow, smirk & dark turtleneck
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<circle cx="24" cy="24" r="23" fill="#EEF1F5" stroke="#475569" stroke-width="2"/>'
        '<path d="M12 42C12.5 35.5 17 32 24 32C31 32 35.5 35.5 36 42" fill="#334155"/>'
        '<rect x="20" y="30.5" width="8" height="4" rx="1.5" fill="#1E293B"/>'
        '<circle cx="24" cy="22" r="10.8" fill="#F6D5B8" stroke="#1E293B" stroke-width="1.6"/>'
        '<path d="M13 20C13 12.5 18 9.5 25 10C31 10.5 34.5 13.5 34.8 19C30 15.5 21 15 13 20Z" fill="#1E293B"/>'
        '<path d="M17 19.6C18.5 18.2 20.5 18.2 21.8 19.2" stroke="#1E293B" stroke-width="1.7" stroke-linecap="round"/>'
        '<line x1="26.5" y1="20.4" x2="31" y2="20" stroke="#1E293B" stroke-width="1.7" stroke-linecap="round"/>'
        '<circle cx="19.5" cy="22.6" r="1.5" fill="#1E293B"/>'
        '<circle cx="28.5" cy="22.6" r="1.5" fill="#1E293B"/>'
        '<path d="M21 27.6C23.5 28.4 26.5 27.4 28 26" stroke="#1E293B" stroke-width="1.7" stroke-linecap="round"/>'
        "</svg>"
    )


def _macro_analyst_svg(size: int) -> str:
    # Sharp strategist with rectangular glasses & forest green blazer
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<circle cx="24" cy="24" r="23" fill="#EBF5EF" stroke="#146C43" stroke-width="2"/>'
        '<path d="M11.5 42.5C12 35.5 16.8 32 24 32C31.2 32 36 35.5 36.5 42.5" fill="#146C43"/>'
        '<path d="M21.5 32.5L24 37.5L26.5 32.5Z" fill="#FFFFFF"/>'
        '<circle cx="24" cy="22" r="10.8" fill="#F5D3B3" stroke="#14532D" stroke-width="1.6"/>'
        '<path d="M13.5 19.5C14 12.5 18 10 24 10C30 10 34 12.5 34.5 19.5C31.5 16 16.5 16 13.5 19.5Z" fill="#1E293B"/>'
        '<rect x="15.8" y="20.2" width="6.6" height="4.4" rx="1.2" fill="#FFFFFF" fill-opacity="0.55" stroke="#14532D" stroke-width="1.4"/>'
        '<rect x="25.6" y="20.2" width="6.6" height="4.4" rx="1.2" fill="#FFFFFF" fill-opacity="0.55" stroke="#14532D" stroke-width="1.4"/>'
        '<line x1="22.4" y1="22.4" x2="25.6" y2="22.4" stroke="#14532D" stroke-width="1.4"/>'
        '<circle cx="19.1" cy="22.4" r="1.1" fill="#14532D"/>'
        '<circle cx="28.9" cy="22.4" r="1.1" fill="#14532D"/>'
        '<path d="M21.2 27.6H26.8" stroke="#14532D" stroke-width="1.6" stroke-linecap="round"/>'
        "</svg>"
    )


def _judge_svg(size: int, accent: str, bg: str) -> str:
    # Impartial cartoon judge character with classic judicial collar & composed expression
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">'
        f'<circle cx="24" cy="24" r="23" fill="{bg}" stroke="{accent}" stroke-width="2"/>'
        f'<path d="M11.5 42.5C12 35.5 16.8 32 24 32C31.2 32 36 35.5 36.5 42.5" fill="{accent}"/>'
        '<path d="M20.5 32.5L23.5 38H21.2L19 33.2Z" fill="#FFFFFF"/>'
        '<path d="M27.5 32.5L24.5 38H26.8L29 33.2Z" fill="#FFFFFF"/>'
        '<circle cx="24" cy="21.8" r="10.2" fill="#F8D8B8" stroke="#1F2937" stroke-width="1.5"/>'
        '<path d="M13 21C12.5 13 17.2 9.5 24 9.5C30.8 9.5 35.5 13 35 21C33.5 17 30 14.8 24 14.8C18 14.8 14.5 17 13 21Z" fill="#E2E8F0" stroke="#64748B" stroke-width="1.2"/>'
        '<circle cx="20" cy="22.2" r="1.4" fill="#1F2937"/>'
        '<circle cx="28" cy="22.2" r="1.4" fill="#1F2937"/>'
        '<path d="M21 27.2C22.6 28.2 25.4 28.2 27 27.2" stroke="#1F2937" stroke-width="1.6" stroke-linecap="round"/>'
        "</svg>"
    )


def get_avatar_svg(entity_id: str, size: int = 38) -> str:
    """Return a clean inline SVG cartoon avatar for any persona or Meta-Judge."""
    key = entity_id.strip().lower()
    if key == "retail_fomo":
        return _retail_fomo_svg(size)
    if key == "fundamentalist":
        return _fundamentalist_svg(size)
    if key == "paranoid_bear":
        return _paranoid_bear_svg(size)
    if key == "contrarian":
        return _contrarian_svg(size)
    if key == "macro_analyst":
        return _macro_analyst_svg(size)
    if key == "judge_1":
        return _judge_svg(size, accent="#16243D", bg="#E8EEF6")
    if key == "judge_2":
        return _judge_svg(size, accent="#0F766E", bg="#E6F4F1")
    if key == "judge_3":
        return _judge_svg(size, accent="#9A5B13", bg="#FDF4E7")
    return _judge_svg(size, accent="#16243D", bg="#EEF1F5")
