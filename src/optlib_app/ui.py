"""Look and feel of the dashboard: palettes, the light/dark switch, page CSS and small HTML pieces.

Streamlit draws its own widgets (tables, inputs) from the theme in .streamlit/config.toml, which defines a light
and a dark palette. The sidebar switch stores the reader's choice the same way Streamlit's own Settings menu
does (browser storage) and reloads the page, so every widget, table and chart changes together.
"""
from __future__ import annotations

import html
import json

import streamlit as st
import streamlit.components.v1 as components

PALETTES = {
    "light": dict(bg="#f6f7fb", card="#ffffff", soft="#f1f4fa", ink="#0f172a", muted="#64748b", line="#e3e8f1",
                  accent="#4f46e5", accent_soft="rgba(79,70,229,0.08)", accent2="#7c3aed", accent_glow="rgba(79,70,229,0.28)", pill1="#4f46e5", pill2="#7c3aed", hero1="#0b1220", hero2="#1e1b4b",
                  shadow="0 1px 2px rgba(15,23,42,.04), 0 4px 14px rgba(15,23,42,.05)"),
    "dark": dict(bg="#0b1220", card="#111a2e", soft="#0f1729", ink="#e2e8f0", muted="#94a3b8", line="#22304a",
                 accent="#818cf8", accent_soft="rgba(129,140,248,0.12)", accent2="#a78bfa", accent_glow="rgba(99,102,241,0.35)", pill1="#4f46e5", pill2="#7c3aed", hero1="#111a2e", hero2="#1e1b4b",
                 shadow="0 1px 2px rgba(0,0,0,.3), 0 6px 18px rgba(0,0,0,.25)"),
}
STATUS = {"PASS": "#16a34a", "WARN": "#d97706", "FAIL": "#dc2626", "SKIP": "#6b7280"}


def embed(html_doc: str, height: int = 0) -> None:
    """Our own HTML/JS in a same-origin iframe (st.iframe on current Streamlit; components.html before it).
    Only ever pass HTML this app generated, never user input."""
    if hasattr(st, "iframe"):
        st.iframe(html_doc, height=max(int(height), 1))
    else:
        components.html(html_doc, height=height, scrolling=height > 0)


def current_theme() -> str:
    try:
        return "dark" if (st.context.theme.type or "light") == "dark" else "light"
    except AttributeError:  # older Streamlit: no theme info, assume light
        return "light"


def theme_switch(where=st.sidebar) -> bool:
    """A Light/Dark switch; returns True when dark. Changing it reloads the page in the other theme."""
    dark_now = current_theme() == "dark"
    dark = where.toggle("Dark mode", value=dark_now, help="Switches every chart and table; the page reloads.")
    if dark != dark_now:
        choice = json.dumps(json.dumps("Dark" if dark else "Light"))
        embed("<script>const p = window.parent; p.localStorage.setItem('stActiveTheme-' + "
                        f"p.location.pathname + '-v2', {choice}); p.location.reload();</script>")
    return dark_now


def inject_css(theme: str) -> None:
    p = PALETTES[theme]
    st.markdown(f"""<style>
:root {{ --bg:{p['bg']}; --card:{p['card']}; --soft:{p['soft']}; --ink:{p['ink']}; --muted:{p['muted']}; --line:{p['line']};
        --accent:{p['accent']}; --accent-soft:{p['accent_soft']}; --shadow:{p['shadow']}; }}
.block-container {{ padding-top: 3.6rem; padding-bottom: 3rem; max-width: 1400px; }}
h1, h2, h3, h4 {{ letter-spacing: -0.01em; }}
[data-testid="stMetric"] {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px;
        padding: 14px 16px 12px; box-shadow: var(--shadow); }}
[data-testid="stMetricLabel"] p {{ font-size: 0.74rem; text-transform: uppercase; letter-spacing: .06em;
        color: var(--muted); font-weight: 600; }}
[data-testid="stMetricValue"] {{ font-size: 1.55rem; font-weight: 700; font-variant-numeric: tabular-nums; }}
/* main navigation: a card of icon tabs, sticky while scrolling, account pinned right.
   (no ">" combinators: Streamlit's markdown escapes them inside <style>) */
.stTabs [role="tablist"] {{ position: sticky; top: 3.9rem; z-index: 60; margin-bottom: 14px; gap: 1px;
        padding: 5px; flex-wrap: nowrap; overflow-x: auto; scrollbar-width: none; background: var(--card); border: 1px solid var(--line); border-radius: 14px;
        box-shadow: var(--shadow), 0 -14px 0 6px var(--bg); }}
.stTabs [role="tablist"]::-webkit-scrollbar {{ display: none; }}
.stTabs [role="tab"], .stTabs [role="tab"][aria-selected="true"] {{ border: none !important; outline-offset: 2px; }}
.stTabs [role="tab"]::before, .stTabs [role="tab"]::after {{ display: none !important; }}
.stTabs .react-aria-SelectionIndicator {{ display: none !important; }}
.stTabs [role="tab"] {{ flex: 0 0 auto; padding: 7px 9px; border-radius: 10px; font-weight: 600; color: var(--muted);
        transition: background .15s ease, color .15s ease, box-shadow .15s ease; }}
.stTabs [role="tab"] p {{ font-size: 0.83rem; font-weight: 600; color: inherit; white-space: nowrap; }}
.stTabs [role="tab"] span[role="img"] {{ font-size: 1.08rem !important; margin-right: 2px; vertical-align: -3px !important; }}
.stTabs [role="tab"]:hover {{ color: var(--ink); background: var(--soft); }}
.stTabs [role="tab"][aria-selected="true"] {{ color: #ffffff;
        background: linear-gradient(135deg, {p['pill1']} 0%, {p['pill2']} 100%);
        box-shadow: 0 1px 2px rgba(0,0,0,.15), 0 6px 16px {p['accent_glow']}; }}
.stTabs [role="tab"]:last-child {{ margin-left: auto; }}
@media (max-width: 1680px) {{ .stTabs [role="tab"]:last-child p {{ font-size: 0 !important; }}
        .stTabs [role="tab"]:last-child p span[role="img"] {{ margin-right: 0; }}
        .stTabs .stTabs [role="tab"]:last-child p {{ font-size: 0.83rem !important; }} }}
/* tabs inside a page stay light: underline style, not sticky */
.stTabs .stTabs [role="tablist"] {{ position: static; margin-bottom: 6px; padding: 0; background: transparent;
        border: none; box-shadow: none; border-bottom: 1px solid var(--line); border-radius: 0; }}
.stTabs .stTabs [role="tab"] {{ border-radius: 8px 8px 0 0; padding: 8px 12px; }}
.stTabs .stTabs [role="tab"]:last-child {{ margin-left: 0; }}
.stTabs .stTabs [role="tab"][aria-selected="true"] {{ color: var(--accent); background: var(--accent-soft);
        box-shadow: inset 0 -2px 0 var(--accent); }}
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {{ background: var(--card); border: 1px solid var(--line);
        border-radius: 12px; padding: 6px; box-shadow: var(--shadow); }}
[data-testid="stExpander"] details {{ border-radius: 12px; border-color: var(--line); background: var(--card); }}
[data-testid="stForm"] {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px; box-shadow: var(--shadow); }}
[data-testid="stSidebar"] {{ border-right: 1px solid var(--line); }}
.opa-hero {{ background: linear-gradient(120deg, {p['hero1']} 0%, {p['hero2']} 100%); border-radius: 16px;
        padding: 20px 24px; margin-bottom: 18px; color: #fff; border: 1px solid var(--line); }}
.opa-hero .k {{ font-size: 12px; letter-spacing: .16em; text-transform: uppercase; color: #a5b4fc; font-weight: 700; }}
.opa-hero .t {{ font-size: 26px; font-weight: 750; margin-top: 4px; letter-spacing: -0.01em; }}
.opa-hero .s {{ font-size: 13.5px; color: #cbd5e1; margin-top: 4px; }}
.opa-chip {{ display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 700;
        margin: 10px 6px 0 0; background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.18); color: #e2e8f0; }}
.opa-pill {{ display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 12px; font-weight: 700;
        border: 1px solid currentColor; }}
.opa-card {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 16px 18px;
        box-shadow: var(--shadow); }}
.opa-card h4 {{ margin: 0 0 6px; font-size: 1rem; }}
.opa-muted {{ color: var(--muted); font-size: 0.86rem; line-height: 1.55; }}
.opa-brand {{ font-weight: 800; font-size: 1.15rem; letter-spacing: -0.01em; }}
.opa-note {{ background: var(--card); border: 1px solid var(--line); border-left: 4px solid var(--accent);
        border-radius: 12px; padding: 16px 18px; box-shadow: var(--shadow); font-size: 0.98rem; line-height: 1.65; }}
.opa-note .meta {{ color: var(--muted); font-size: 0.78rem; margin-top: 8px; }}
.opa-kicker {{ font-size: 0.72rem; letter-spacing: .12em; text-transform: uppercase; color: var(--accent);
        font-weight: 800; margin-bottom: 6px; }}
.opa-news {{ padding: 10px 2px; border-bottom: 1px solid var(--line); }}
.opa-news a {{ color: var(--ink); text-decoration: none; font-weight: 600; line-height: 1.45; }}
.opa-news a:hover {{ color: var(--accent); }}
.opa-news .src {{ color: var(--muted); font-size: 0.8rem; margin-top: 2px; }}
.opa-brand span {{ color: var(--accent); }}
</style>""", unsafe_allow_html=True)


def pill(status: str) -> str:
    c = STATUS.get(status, "#6b7280")
    return f'<span class="opa-pill" style="color:{c}">{html.escape(status)}</span>'


def hero(kicker: str, title: str, subtitle: str, chips: list[str]) -> None:
    chip_html = "".join(f'<span class="opa-chip">{c}</span>' for c in chips)
    st.markdown(f'<div class="opa-hero"><div class="k">{html.escape(kicker)}</div><div class="t">{html.escape(title)}'
                f'</div><div class="s">{html.escape(subtitle)}</div>{chip_html}</div>', unsafe_allow_html=True)


def card(title: str, body_html: str) -> None:
    st.markdown(f'<div class="opa-card"><h4>{html.escape(title)}</h4><div class="opa-muted">{body_html}</div></div>',
                unsafe_allow_html=True)


def news_list(items: list[dict], ago=None) -> None:
    """Headlines as links (source and age underneath)."""
    rows = []
    for n in items:
        when = f" · {ago(n.get('published'))}" if ago and n.get("published") else ""
        rows.append(f'<div class="opa-news"><a href="{html.escape(n["link"])}" target="_blank" rel="noopener">'
                    f'{html.escape(n["title"])}</a><div class="src">{html.escape(n["source"])}{when}</div></div>')
    st.markdown("".join(rows), unsafe_allow_html=True)


def note_card(text: str, meta: str) -> None:
    st.markdown(f'<div class="opa-note"><div class="opa-kicker">AI market note</div>{html.escape(text)}'
                f'<div class="meta">{html.escape(meta)}</div></div>', unsafe_allow_html=True)
