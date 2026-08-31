"""
Popola la knowledge base con il cluster 'navigation UI'.

Nodi seminati (pilota iniziale, ~20 nodi curati):
  CSSProperty:  position, top, left, right, bottom, z-index, transform,
                transition, display, overflow
  HTMLElement:  nav, header, button, dialog, details
  AriaRole:     navigation, menubar, menu, menuitem
  AriaAttribute: aria-expanded, aria-haspopup, aria-hidden, aria-label
  Pattern:      sticky-header, hamburger-menu, dropdown-menu, sidebar-drawer
  Technique:    focus-trap, keyboard-navigation, click-outside-close
  Problem:      inaccessible-menu-keyboard, layout-shift-on-scroll
  Solution:     use-position-sticky, aria-expanded-toggle
  Example:      nav-fixed-top-example, hamburger-menu-react

Relazioni: ~50-80 relazioni tipizzate tra questi nodi.

Il contenuto testuale ('content') dei nodi CSSProperty viene estratto dai
file MDN già cachati in rag_system/manuals_cache/, non riscritto da zero.

Uso:
    python3 seed_navigation.py            # popola + stampa stats
    python3 seed_navigation.py --clear    # svuota prima di ripopolare
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from kb import KnowledgeBase


MDN_CACHE = Path(__file__).parent.parent / "rag_system" / "manuals_cache"


def load_mdn_summary(prop: str, max_chars: int = 800) -> tuple[str, str]:
    """
    Legge un file MDN cachato, ritorna (summary_1frase, content_snippet).
    Se il file non esiste, ritorna placeholder.
    """
    slug = prop.replace("@", "at-").replace(":", "").replace("/", "_")
    path = MDN_CACHE / f"mdn_css_{slug}.txt"
    if not path.exists():
        return f"CSS property `{prop}`", f"See MDN documentation for `{prop}`."
    text = path.read_text(errors="ignore")
    # rimuovi il titolo (prima riga con #) e prendi il primo paragrafo come summary
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    paragraphs = [l for l in lines if not l.startswith("#") and len(l) > 30]
    summary = (paragraphs[0][:200] + "…") if paragraphs else f"CSS property `{prop}`"
    # content = primi max_chars di paragrafi utili
    content = "\n\n".join(paragraphs[:5])[:max_chars]
    return summary, content


# ═══ NODI: CSS Properties (contenuto da MDN cache) ═══════════════════════════
CSS_PROPERTIES = [
    ("css-position",  "position",  ["static", "relative", "absolute", "fixed", "sticky"]),
    ("css-top",       "top",       ["length", "percentage", "auto"]),
    ("css-left",      "left",      ["length", "percentage", "auto"]),
    ("css-right",     "right",     ["length", "percentage", "auto"]),
    ("css-bottom",    "bottom",    ["length", "percentage", "auto"]),
    ("css-z-index",   "z-index",   ["integer", "auto"]),
    ("css-transform", "transform", ["translate()", "rotate()", "scale()", "matrix()"]),
    ("css-transition","transition",["property", "duration", "timing-function", "delay"]),
    ("css-display",   "display",   ["block", "flex", "grid", "none", "inline-block"]),
    ("css-overflow",  "overflow",  ["visible", "hidden", "scroll", "auto"]),
]


# ═══ NODI: HTML Elements ═════════════════════════════════════════════════════
HTML_ELEMENTS = [
    ("html-nav", "nav",
     "The HTML <nav> element represents a section of a page whose purpose is to provide navigation links.",
     "Use <nav> for major navigation blocks. Not every group of links needs <nav>; only primary/site navigation. Screen readers use it as a landmark."),
    ("html-header", "header",
     "The HTML <header> element represents introductory content, typically a group of introductory or navigational aids.",
     "Common placement for site brand + primary navigation. Multiple <header> allowed (one per section). Not the same as <head>."),
    ("html-button", "button",
     "The HTML <button> element is an interactive element activated by a user with a mouse, keyboard, finger, voice command, or other assistive technology.",
     "Always prefer <button> over <div onClick> for interactive triggers — provides keyboard, ARIA and focus semantics for free. Use type='button' inside <form> to avoid accidental submit."),
    ("html-dialog", "dialog",
     "The HTML <dialog> element represents a modal or non-modal dialog box or other interactive component.",
     "Native modal management with .showModal() traps focus and adds ::backdrop. Wide browser support since 2022. Preferred over custom modal for most cases."),
    ("html-details", "details",
     "The HTML <details> element creates a disclosure widget in which information is visible only when the widget is toggled into an 'open' state.",
     "Zero-JS disclosure: paired with <summary>. Perfect for FAQ, mobile hamburger fallback, sidebar collapse. Free keyboard/screen-reader support."),
]


# ═══ NODI: ARIA Roles ════════════════════════════════════════════════════════
ARIA_ROLES = [
    ("aria-role-navigation", "role='navigation'",
     "A landmark identifying a section for navigating a document or set of documents.",
     "Implicit on <nav>. Adding role='navigation' to a <div> works but <nav> is preferred. Screen readers list all navigation landmarks in a rotor menu."),
    ("aria-role-menubar", "role='menubar'",
     "A presentation of menu that usually remains visible and is usually presented horizontally.",
     "For persistent app-like menu bars (like OS menus). Requires full keyboard model: arrows to move between items, Enter to activate. Do NOT use for site navigation — use <nav> instead."),
    ("aria-role-menu", "role='menu'",
     "A type of widget that offers a list of choices to the user.",
     "For actions in a menu, not for navigation links. Requires proper keyboard: arrows to move, Escape to close, focus management. Complex to implement correctly."),
    ("aria-role-menuitem", "role='menuitem'",
     "An option in a set of choices contained by a menu or menubar.",
     "Child of role='menu' or role='menubar'. Must handle arrow keys, activation on Enter/Space, and focus restoration when menu closes."),
]


# ═══ NODI: ARIA Attributes ═══════════════════════════════════════════════════
ARIA_ATTRIBUTES = [
    ("aria-attr-expanded", "aria-expanded",
     "Indicates whether a grouping element owned or controlled by this element is expanded or collapsed.",
     "Set on the TRIGGER (button), not on the menu itself. Toggle between 'true' and 'false' on click. Essential for hamburger, dropdowns, accordions."),
    ("aria-attr-haspopup", "aria-haspopup",
     "Indicates the availability and type of interactive popup element that can be triggered by this element.",
     "Values: 'menu' | 'listbox' | 'tree' | 'grid' | 'dialog' | 'true' (deprecated, means menu). Announces to screen reader that a popup follows the trigger."),
    ("aria-attr-hidden", "aria-hidden",
     "Indicates whether the element is exposed to an accessibility API.",
     "Use 'true' to hide decorative elements from screen readers. NEVER put aria-hidden on focusable elements — creates focus trap for AT users."),
    ("aria-attr-label", "aria-label",
     "Defines a string value that labels the current element.",
     "For elements without visible text (icon-only buttons). Icon hamburger button MUST have aria-label='Open menu' or similar."),
]


# ═══ NODI: Pattern UI ═══════════════════════════════════════════════════════
PATTERNS = [
    ("pattern-sticky-header",
     "Pattern",
     "Sticky Header (sticks to top on scroll)",
     "Header that scrolls with content until it reaches the top, then sticks in place while user continues scrolling.",
     "Two implementation choices: `position: sticky; top: 0` (scoped to parent, no layout shift, one line CSS) or `position: fixed; top: 0` (removed from flow, need padding compensation on body/main to avoid content hiding under header). Sticky is preferred for header of a scrolling section; fixed for global always-visible header.",
     """/* Preferred: position: sticky */
.header {
  position: sticky;
  top: 0;
  z-index: 100;
  background: white;
}

/* Alternative: position: fixed (needs body padding) */
.header--fixed { position: fixed; top: 0; left: 0; right: 0; z-index: 100; }
body { padding-top: 64px; /* height of fixed header */ }"""),

    ("pattern-hamburger-menu",
     "Pattern",
     "Hamburger Menu (mobile navigation toggle)",
     "Icon button (3 lines) that toggles visibility of the navigation menu on small screens.",
     "Consists of: (1) a <button> with aria-label, aria-expanded, aria-controls; (2) a <nav> or <ul> menu that gets shown/hidden via CSS class controlled by a state variable. On mobile the menu is hidden by default; on desktop the button is display:none and the menu is always visible. Focus should be managed: when opened, focus moves to first item; Escape closes the menu.",
     """// React version
function Nav() {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button
        aria-label={open ? "Close menu" : "Open menu"}
        aria-expanded={open}
        aria-controls="mobile-nav"
        onClick={() => setOpen(!open)}
        className="hamburger"
      >
        <span /><span /><span />
      </button>
      <nav id="mobile-nav" className={open ? "nav open" : "nav"}>
        <a href="/">Home</a>
        <a href="/about">About</a>
      </nav>
    </>
  );
}"""),

    ("pattern-dropdown-menu",
     "Pattern",
     "Dropdown Menu (menu appears on click/hover)",
     "Menu of links or actions that appears below a trigger button, positioned relative to it.",
     "Use aria-expanded on the trigger, aria-controls to link to the menu. Handle click-outside to close, Escape key to close and return focus to trigger. Position with `position: absolute; top: 100%; left: 0` relative to a parent with `position: relative`.",
     """function Dropdown({ trigger, items }) {
  const [open, setOpen] = useState(false);
  const ref = useRef();
  useEffect(() => {
    if (!open) return;
    const onDoc = (e) => { if (!ref.current?.contains(e.target)) setOpen(false); };
    document.addEventListener('mousedown', onDoc);
    return () => document.removeEventListener('mousedown', onDoc);
  }, [open]);
  return (
    <div ref={ref} className="dropdown-wrap">
      <button aria-expanded={open} aria-haspopup="menu" onClick={() => setOpen(!open)}>
        {trigger}
      </button>
      {open && <ul role="menu">{items.map(i => <li role="menuitem" key={i}>{i}</li>)}</ul>}
    </div>
  );
}"""),

    ("pattern-sidebar-drawer",
     "Pattern",
     "Sidebar / Slide-in Drawer",
     "Panel that slides in from the side of the viewport, typically for navigation on mobile or secondary content on desktop.",
     "Common implementation: fixed-position panel translated off-screen (transform: translateX(-100%)), toggled by adding a class that resets translate to 0. Combine with a backdrop <div> that covers the rest of the screen to indicate modality. Focus should be trapped inside while open, Escape closes.",
     """.drawer {
  position: fixed;
  top: 0; left: 0; bottom: 0;
  width: 280px;
  transform: translateX(-100%);
  transition: transform 200ms ease-out;
  z-index: 200;
}
.drawer--open { transform: translateX(0); }
.backdrop { position: fixed; inset: 0; background: rgba(0,0,0,0.4); z-index: 150; }"""),
]


# ═══ NODI: Techniques ═══════════════════════════════════════════════════════
TECHNIQUES = [
    ("tech-focus-trap",
     "Technique",
     "Focus Trap (keep tab focus inside a container)",
     "Prevent keyboard focus from leaving a modal/dialog/drawer while it is open, so Tab cycles within it.",
     "For native <dialog> with .showModal() the browser does this automatically. For custom implementations: find all focusable elements inside the container, listen for Tab (forward) / Shift+Tab (back), redirect focus to first/last when at boundaries. Restore focus to trigger when closing.",
     """function trapFocus(container) {
  const focusables = container.querySelectorAll(
    'a, button, input, textarea, select, [tabindex]:not([tabindex="-1"])'
  );
  const first = focusables[0];
  const last = focusables[focusables.length - 1];
  container.addEventListener('keydown', (e) => {
    if (e.key !== 'Tab') return;
    if (e.shiftKey && document.activeElement === first) {
      last.focus(); e.preventDefault();
    } else if (!e.shiftKey && document.activeElement === last) {
      first.focus(); e.preventDefault();
    }
  });
}"""),

    ("tech-keyboard-navigation",
     "Technique",
     "Keyboard Navigation for Menus",
     "Support arrow keys, Home, End, Escape, and Enter/Space for navigating menu items with keyboard.",
     "Roving tabindex pattern: only one item in the menu has tabindex=0 at any time, others have tabindex=-1. Arrow keys shift which item has tabindex=0 and move focus. This lets Tab escape the widget (goes to next form element on the page) while arrows navigate inside.",
     """function handleMenuKeydown(e, items, currentIdx, setIdx) {
  switch(e.key) {
    case 'ArrowDown': setIdx((currentIdx + 1) % items.length); e.preventDefault(); break;
    case 'ArrowUp':   setIdx((currentIdx - 1 + items.length) % items.length); e.preventDefault(); break;
    case 'Home':      setIdx(0); e.preventDefault(); break;
    case 'End':       setIdx(items.length - 1); e.preventDefault(); break;
    case 'Escape':    close(); e.preventDefault(); break;
  }
}"""),

    ("tech-click-outside-close",
     "Technique",
     "Close on Click Outside",
     "Close a dropdown/popup when the user clicks anywhere outside of it.",
     "Attach a mousedown listener on document, check if event.target is inside the popup element (via ref.current.contains). If not, close the popup. Add listener only when popup is open, remove in cleanup. Alternative: <dialog> with light dismiss via clicking outside .showModal()'s backdrop area.",
     """useEffect(() => {
  if (!open) return;
  function onClickOutside(e) {
    if (ref.current && !ref.current.contains(e.target)) {
      setOpen(false);
    }
  }
  document.addEventListener('mousedown', onClickOutside);
  return () => document.removeEventListener('mousedown', onClickOutside);
}, [open]);"""),
]


# ═══ NODI: Problems & Solutions ═════════════════════════════════════════════
PROBLEMS_SOLUTIONS = [
    ("problem-menu-keyboard-inaccessible",
     "Problem",
     "Menu inaccessible via keyboard",
     "Users navigating with keyboard (Tab, arrows) cannot open, navigate, or close a custom dropdown/menu.",
     "Symptoms: button doesn't respond to Enter/Space; menu items are not reachable; Escape doesn't close menu; focus is 'lost' after closing. Fails WCAG 2.1.1 (Keyboard). Extremely common on custom-built menus with <div>+onClick that skip the semantic layer."),
    ("problem-layout-shift-on-fixed-header",
     "Problem",
     "Layout shift when adding position:fixed to a header",
     "When header changes from static to fixed, the content below jumps up because the header is removed from the document flow.",
     "The fix is to add matching padding-top (or margin-top) to the next-in-flow element equal to the header's height. Alternatively, `position: sticky` avoids this entirely because sticky doesn't remove the element from flow."),

    ("solution-use-native-button",
     "Solution",
     "Use <button> instead of <div onClick>",
     "The native <button> element handles keyboard activation, focus, ARIA, and disabled state for free.",
     "Every interactive trigger (menu toggle, dropdown opener, close button) should be a <button>. Style with CSS to look however you want. Add type='button' if inside a <form> to prevent accidental submit."),
    ("solution-use-position-sticky",
     "Solution",
     "Use position: sticky instead of position: fixed for headers",
     "Sticky doesn't remove the element from flow, so no layout shift and no manual padding compensation needed.",
     "Trade-off: sticky is scoped to its parent's scrolling container — if you want the header to be sticky across the whole page, ensure its parent is <body> (or a full-height scrolling container). Fixed still needed if you want the header to overlay content that scrolls behind it."),
]


# ═══ RELAZIONI ══════════════════════════════════════════════════════════════
# Formato: (source_id, RELATION, target_id, optional_note)
RELATIONS = [
    # --- Pattern USES CSS ---
    ("pattern-sticky-header", "USES", "css-position", "specifically position: sticky or fixed"),
    ("pattern-sticky-header", "USES", "css-top", "top: 0 to anchor at viewport top"),
    ("pattern-sticky-header", "USES", "css-z-index", "to layer above scrolling content"),
    ("pattern-hamburger-menu", "USES", "css-display", "toggle between block and none for mobile menu"),
    ("pattern-hamburger-menu", "USES", "css-transition", "smooth reveal animation"),
    ("pattern-dropdown-menu", "USES", "css-position", "position: absolute for the menu relative to trigger"),
    ("pattern-dropdown-menu", "USES", "css-top", "top: 100% to place menu below trigger"),
    ("pattern-dropdown-menu", "USES", "css-z-index", "to layer above surrounding content"),
    ("pattern-sidebar-drawer", "USES", "css-position", "position: fixed to overlay viewport"),
    ("pattern-sidebar-drawer", "USES", "css-transform", "translateX for slide animation"),
    ("pattern-sidebar-drawer", "USES", "css-transition", "animate the transform smoothly"),
    ("pattern-sidebar-drawer", "USES", "css-z-index", "must stack above main content"),

    # --- Pattern USES HTML ---
    ("pattern-sticky-header", "USES", "html-header", "the semantic container"),
    ("pattern-sticky-header", "USES", "html-nav", "for primary navigation inside header"),
    ("pattern-hamburger-menu", "USES", "html-button", "MUST be a real button, not a div"),
    ("pattern-hamburger-menu", "USES", "html-nav", "the menu itself should be a nav"),
    ("pattern-dropdown-menu", "USES", "html-button", "the trigger must be a button"),
    ("pattern-sidebar-drawer", "USES", "html-dialog", "native dialog can be used for modal drawers"),

    # --- Pattern USES ARIA ---
    ("pattern-hamburger-menu", "USES", "aria-attr-expanded", "on the toggle button"),
    ("pattern-hamburger-menu", "USES", "aria-attr-label", "for the icon-only button"),
    ("pattern-hamburger-menu", "USES", "aria-role-navigation", "implicit on <nav>"),
    ("pattern-dropdown-menu", "USES", "aria-attr-expanded", "state of the trigger"),
    ("pattern-dropdown-menu", "USES", "aria-attr-haspopup", "signals popup follows"),
    ("pattern-sidebar-drawer", "USES", "aria-attr-hidden", "when drawer is closed"),

    # --- Pattern COMPOSES / REQUIRES ---
    ("pattern-hamburger-menu", "COMPOSES", "pattern-dropdown-menu", "conceptually a dropdown triggered by hamburger icon"),
    ("pattern-hamburger-menu", "REQUIRES", "tech-click-outside-close", "for proper UX"),
    ("pattern-hamburger-menu", "REQUIRES", "tech-keyboard-navigation", "for accessibility"),
    ("pattern-dropdown-menu", "REQUIRES", "tech-click-outside-close"),
    ("pattern-dropdown-menu", "REQUIRES", "tech-keyboard-navigation"),
    ("pattern-sidebar-drawer", "REQUIRES", "tech-focus-trap", "focus must not escape open drawer"),
    ("pattern-sidebar-drawer", "REQUIRES", "tech-keyboard-navigation", "Escape should close"),

    # --- Alternatives ---
    ("pattern-sticky-header", "ALTERNATIVE_TO", "pattern-hamburger-menu", "different responses to limited vertical space"),
    ("html-details", "ALTERNATIVE_TO", "pattern-dropdown-menu", "zero-JS alternative when accessibility of native disclosure suffices"),
    ("html-dialog", "ALTERNATIVE_TO", "pattern-sidebar-drawer", "when the drawer should be modal, prefer native <dialog>"),

    # --- Solutions SOLVE Problems ---
    ("solution-use-native-button", "SOLVES", "problem-menu-keyboard-inaccessible", "native button provides keyboard activation for free"),
    ("solution-use-position-sticky", "SOLVES", "problem-layout-shift-on-fixed-header", "sticky stays in flow, no compensation needed"),

    # --- Solutions USE things ---
    ("solution-use-native-button", "USES", "html-button"),
    ("solution-use-position-sticky", "USES", "css-position"),

    # --- Techniques USES ---
    ("tech-focus-trap", "USES", "aria-attr-hidden", "sibling elements outside modal should be aria-hidden while open"),

    # --- Related / See also ---
    ("aria-role-menubar", "SEE_ALSO", "aria-role-menu", "do NOT confuse: menubar is persistent, menu is transient"),
    ("aria-role-navigation", "SEE_ALSO", "html-nav", "implicit role on <nav>"),
    ("pattern-sticky-header", "SEE_ALSO", "problem-layout-shift-on-fixed-header", "the reason to prefer sticky over fixed"),
    ("css-position", "SEE_ALSO", "css-top", "top/left/right/bottom only apply to positioned elements"),
    ("css-transform", "SEE_ALSO", "css-transition", "commonly combined for animated transforms"),
]


# ═══ MAIN ════════════════════════════════════════════════════════════════════
def build_nodes() -> list[dict]:
    nodes = []

    # CSS Properties (contenuto da MDN cache)
    for node_id, prop, values in CSS_PROPERTIES:
        summary, content = load_mdn_summary(prop)
        nodes.append({
            "id": node_id, "type": "CSSProperty",
            "title": f"CSS property: {prop}",
            "summary": summary,
            "content": content,
            "source": f"https://developer.mozilla.org/en-US/docs/Web/CSS/{prop.lstrip('@')}",
            "concepts_used": values,
            "tags": [prop, "css", "styling"],
        })

    # HTML Elements
    for node_id, tag, summary, content in HTML_ELEMENTS:
        nodes.append({
            "id": node_id, "type": "HTMLElement",
            "title": f"HTML element: <{tag}>",
            "summary": summary,
            "content": content,
            "source": f"https://developer.mozilla.org/en-US/docs/Web/HTML/Element/{tag}",
            "tags": [tag, "html", "semantic"],
        })

    # ARIA Roles
    for node_id, role, summary, content in ARIA_ROLES:
        nodes.append({
            "id": node_id, "type": "AriaRole",
            "title": role,
            "summary": summary,
            "content": content,
            "source": f"https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/Roles/{role.replace(chr(39), '').replace('role=', '')}_role",
            "tags": ["aria", "accessibility", "role"],
        })

    # ARIA Attributes
    for node_id, attr, summary, content in ARIA_ATTRIBUTES:
        nodes.append({
            "id": node_id, "type": "AriaAttribute",
            "title": attr,
            "summary": summary,
            "content": content,
            "source": f"https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/Attributes/{attr}",
            "tags": ["aria", "accessibility", "attribute"],
        })

    # Patterns
    for node_id, node_type, title, summary, content, code in PATTERNS:
        nodes.append({
            "id": node_id, "type": node_type,
            "title": title, "summary": summary, "content": content,
            "code_snippet": code,
            "tags": ["pattern", "navigation", "ui"],
        })

    # Techniques
    for node_id, node_type, title, summary, content, code in TECHNIQUES:
        nodes.append({
            "id": node_id, "type": node_type,
            "title": title, "summary": summary, "content": content,
            "code_snippet": code,
            "tags": ["technique", "accessibility", "interaction"],
        })

    # Problems & Solutions
    for tup in PROBLEMS_SOLUTIONS:
        node_id, node_type, title, summary, content = tup
        nodes.append({
            "id": node_id, "type": node_type,
            "title": title, "summary": summary, "content": content,
            "tags": [node_type.lower(), "navigation"],
        })

    return nodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clear", action="store_true", help="clear DB before seeding")
    args = ap.parse_args()

    kb = KnowledgeBase()
    print("[main] initializing schema...", flush=True)
    kb.init_schema()

    if args.clear:
        print("[main] clearing existing data...", flush=True)
        kb.clear()

    nodes = build_nodes()
    print(f"[main] inserting {len(nodes)} nodes...", flush=True)
    for n in nodes:
        try:
            kb.add_node(n)
        except Exception as e:
            print(f"  [error] node {n.get('id')}: {e}", flush=True)

    print(f"[main] inserting {len(RELATIONS)} relations...", flush=True)
    for rel in RELATIONS:
        src, rel_type, dst = rel[0], rel[1], rel[2]
        note = rel[3] if len(rel) > 3 else None
        try:
            kb.add_relation(src, rel_type, dst, note)
        except Exception as e:
            print(f"  [error] rel {src}-{rel_type}->{dst}: {e}", flush=True)

    print("\n=== STATS ===")
    stats = kb.stats()
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Total relations: {stats['total_relations']}")
    print(f"By type: {stats['nodes_by_type']}")
    print(f"By relation: {stats['relations_by_type']}")

    kb.close()


if __name__ == "__main__":
    main()
