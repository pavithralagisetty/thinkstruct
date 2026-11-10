"""CPC classification codes.

The data stores codes without separators: "B60C1112FI" is CPC B60C 11/12 (the trailing
"FI" is not part of the symbol). The digits alone are ambiguous, 1112 could be 1/112 or
11/12, so codes are split with the real main groups of each subclass.
"""

import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# Main groups per subclass, without the indexing codes (2000 series). Generated from the USPTO
# CPC scheme pages https://www.uspto.gov/web/patents/classification/cpc/html/cpc-<SUBCLASS>.html
# (fetched 2026-11-09). Titles are the scheme titles without cross references in brackets.
MAIN_GROUPS: dict[str, frozenset[int]] = {
    "B60B": frozenset({1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25, 27, 29, 30, 31, 33, 35, 37, 39}),
    "B60C": frozenset({1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 23, 25, 27, 29, 99}),
    "B60D": frozenset({1, 3, 5, 99}),
    "B60F": frozenset({1, 3, 5}),
    "B60G": frozenset({1, 3, 5, 7, 9, 11, 13, 15, 17, 21, 99}),
    "B60J": frozenset({1, 3, 5, 7, 9, 10, 11}),
    "B60R": frozenset({1, 3, 5, 7, 9, 11, 13, 15, 16, 17, 19, 21, 22, 25, 99}),
}

SUBCLASS_TITLES: dict[str, str] = {
    "B60B": "VEHICLE WHEELS; CASTORS; AXLES FOR WHEELS OR CASTORS; INCREASING WHEEL ADHESION",
    "B60C": "VEHICLE TYRES; TYRE INFLATION; TYRE CHANGING; CONNECTING VALVES TO INFLATABLE ELASTIC BODIES IN GENERAL; DEVICES OR ARRANGEMENTS RELATED TO TYRES",
    "B60D": "VEHICLE CONNECTIONS",
    "B60F": "VEHICLES FOR USE BOTH ON RAIL AND ON ROAD; VEHICLES CAPABLE OF TRAVELLING IN OR ON DIFFERENT MEDIA, e.g. AMPHIBIOUS VEHICLES",
    "B60G": "VEHICLE SUSPENSION ARRANGEMENTS",
    "B60J": "WINDOWS, WINDSCREENS, NON-FIXED ROOFS, DOORS, OR SIMILAR DEVICES FOR VEHICLES; REMOVABLE EXTERNAL PROTECTIVE COVERINGS SPECIALLY ADAPTED FOR VEHICLES (fastening, suspending, closing, or opening of such devices E05)",
    "B60R": "VEHICLES, VEHICLE FITTINGS, OR VEHICLE PARTS, NOT OTHERWISE PROVIDED FOR",
}

GROUP_TITLES: dict[tuple[str, int], str] = {
    ("B60B", 1): "Spoked wheels; Spokes thereof",
    ("B60B", 3): "Disc wheels, i.e. wheels with load-supporting disc body",
    ("B60B", 5): "Wheels, spokes, disc bodies, rims, hubs, wholly or predominantly made of non-metallic material",
    ("B60B", 7): "Wheel cover discs, rings, or the like, for ornamenting, protecting, venting, or obscuring, wholly or in part, the wheel body, rim, hub, or tyre sidewall, e.g. wheel cover discs, wheel cover discs with cooling fins",
    ("B60B", 9): "Wheels of high resiliency, e.g. with conical interacting pressure-surfaces",
    ("B60B", 11): "Units comprising multiple wheels arranged side by side; Wheels having more than one rim or capable of carrying more than one tyre",
    ("B60B", 15): "Wheels or wheel attachments designed for increasing traction",
    ("B60B", 17): "Wheels characterised by rail-engaging elements",
    ("B60B", 19): "Wheels not otherwise provided for or having characteristics specified in one of the subgroups of this group",
    ("B60B", 21): "Rims",
    ("B60B", 23): "Attaching rim to wheel body",
    ("B60B", 25): "Rims built-up of several main parts; Locking means for the rim parts",
    ("B60B", 27): "Hubs",
    ("B60B", 29): "Apparatus or tools for mounting or dismounting wheels",
    ("B60B", 30): "Means for holding wheels or parts thereof",
    ("B60B", 31): "Apparatus or tools for assembling or disassembling wheels",
    ("B60B", 33): "Castors in general; Anti-clogging castors",
    ("B60B", 35): "Axle units; Parts thereof; Arrangements for lubrication of axles",
    ("B60B", 37): "Wheel-axle combinations, e.g. wheel sets",
    ("B60B", 39): "Increasing wheel adhesion",
    ("B60C", 1): "Tyres characterised by the chemical composition or the physical arrangement or mixture of the composition",
    ("B60C", 3): "Tyres characterised by the transverse section",
    ("B60C", 5): "Inflatable pneumatic tyres or inner tubes",
    ("B60C", 7): "Non-inflatable or solid tyres",
    ("B60C", 9): "Reinforcements or ply arrangement of pneumatic tyres",
    ("B60C", 11): "Tyre tread bands; Tread patterns; Anti-skid inserts",
    ("B60C", 13): "Tyre sidewalls; Protecting, decorating, marking, or the like, thereof",
    ("B60C", 15): "Tyre beads, e.g. ply turn-up or overlap",
    ("B60C", 17): "Tyres characterised by means enabling restricted operation in damaged or deflated condition; Accessories therefor",
    ("B60C", 19): "Tyre parts or constructions not otherwise provided for",
    ("B60C", 23): "Devices for measuring, signalling, controlling, or distributing tyre pressure or temperature, specially adapted for mounting on vehicles; Arrangement of tyre inflating devices on vehicles, e.g. of pumps or of tanks; Tyre cooling arrangements",
    ("B60C", 25): "Apparatus or tools adapted for mounting, removing or inspecting tyres",
    ("B60C", 27): "Non-skid devices temporarily attachable to resilient tyres or resiliently-tyred wheels",
    ("B60C", 29): "Arrangements of tyre-inflating valves to tyres or rims; Accessories for tyre-inflating valves, not otherwise provided for",
    ("B60C", 99): "Subject matter not provided for in other groups of this subclass",
    ("B60D", 1): "Traction couplings; Hitches; Draw-gear; Towing devices",
    ("B60D", 3): "Fittings to facilitate pushing",
    ("B60D", 5): "Gangways for coupled vehicles, e.g. of concertina type",
    ("B60D", 99): "Subject matter not provided for in other groups of this subclass",
    ("B60F", 1): "Vehicles for use both on rail and on road; Conversions therefor",
    ("B60F", 3): "Amphibious vehicles, i.e. vehicles capable of travelling both on land and on water; Land vehicles capable of travelling under water",
    ("B60F", 5): "Other vehicles capable of travelling in or on different media",
    ("B60G", 1): "Suspensions with rigid connection between axle and frame",
    ("B60G", 3): "Resilient suspensions for a single wheel",
    ("B60G", 5): "Resilient suspensions for a set of tandem wheels or axles having interrelated movements",
    ("B60G", 7): "Pivoted suspension arms; Accessories thereof",
    ("B60G", 9): "Resilient suspensions of a rigid axle or axle housing for two or more wheels",
    ("B60G", 11): "Resilient suspensions characterised by arrangement, location or kind of springs",
    ("B60G", 13): "Resilient suspensions characterised by arrangement, location or type of vibration dampers",
    ("B60G", 15): "Resilient suspensions characterised by arrangement, location or type of combined spring and vibration damper, e.g. telescopic type",
    ("B60G", 17): "Resilient suspensions having means for adjusting the spring or vibration-damper characteristics, for regulating the distance between a supporting surface and a sprung part of vehicle or for locking suspension during use to meet varying vehicular or surface conditions, e.g. due to speed or load",
    ("B60G", 21): "Interconnection systems for two or more resiliently-suspended wheels, e.g. for stabilising a vehicle body with respect to acceleration, deceleration or centrifugal forces",
    ("B60G", 99): "Subject matter not provided for in other groups of this subclass",
    ("B60J", 1): "Windows; Windscreens; Accessories therefor",
    ("B60J", 3): "Antiglare equipment associated with windows or windscreens; Sun visors for vehicles",
    ("B60J", 5): "Doors",
    ("B60J", 7): "Non-fixed roofs; Roofs with movable panels, e.g. rotary sunroofs",
    ("B60J", 9): "Devices not provided for in one of main groups B60J 1/00 - B60J 7/00",
    ("B60J", 10): "Sealing arrangements",
    ("B60J", 11): "Removable external protective coverings specially adapted for vehicles or parts of vehicles, e.g. parking covers",
    ("B60R", 1): "Optical viewing arrangements; Real-time viewing arrangements for drivers or passengers using optical image capturing systems, e.g. cameras or video systems specially adapted for use in or on vehicles",
    ("B60R", 3): "Arrangements of steps or ladders facilitating access to or on the vehicle, e.g. running-boards",
    ("B60R", 5): "Compartments within vehicle body primarily intended or sufficiently spacious for trunks, suit-cases, or the like",
    ("B60R", 7): "Stowing or holding appliances inside vehicle primarily intended for personal property smaller than suit-cases, e.g. travelling articles, or maps",
    ("B60R", 9): "Supplementary fittings on vehicle exterior for carrying loads, e.g. luggage, sports gear or the like",
    ("B60R", 11): "Arrangements for holding or mounting articles, not otherwise provided for",
    ("B60R", 13): "Elements for body-finishing, identifying, or decorating; Arrangements or adaptations for advertising purposes",
    ("B60R", 15): "Arrangements or adaptations of sanitation devices",
    ("B60R", 16): "Electric or fluid circuits specially adapted for vehicles and not otherwise provided for; Arrangement of elements of electric or fluid circuits specially adapted for vehicles and not otherwise provided for",
    ("B60R", 17): "Arrangements or adaptations of lubricating systems or devices",
    ("B60R", 19): "Wheel guards; Radiator guards, e.g. grilles; Obstruction removers; Fittings damping bouncing force in collisions",
    ("B60R", 21): "Arrangements or fittings on vehicles for protecting or preventing injuries to occupants or pedestrians in case of accidents or other traffic risks",
    ("B60R", 22): "Safety belts or body harnesses in vehicles",
    ("B60R", 25): "Fittings or systems for preventing or indicating unauthorised use or theft of vehicles",
    ("B60R", 99): "Subject matter not provided for in other groups of this subclass",
}

MIN_SUBGROUP_DIGITS = 2
_CODE_RE = re.compile(r"^([A-H]\d\d[A-Z])(\d*)([A-Z]*)$")
_QUERY_RE = re.compile(r"^([A-H]\d\d[A-Z])(?:\s*(\d+)(?:\s*/\s*(\d+))?)?$")


@dataclass(frozen=True)
class Cpc:
    subclass: str
    group: int | None = None
    subgroup: str | None = None        # digits after the slash, leading zeros kept; None = whole group

    def __str__(self) -> str:
        if self.group is None:
            return self.subclass
        return f"{self.subclass} {self.group}/{self.subgroup or '00'}"


def _readings(digits: str) -> list[tuple[int, str]]:
    """Every (main group, subgroup) split that leaves a subgroup of at least 2 digits."""
    return [(int(digits[:i]), digits[i:]) for i in range(1, len(digits) - MIN_SUBGROUP_DIGITS + 1)
            if digits[0] != "0"]


def _split_digits(subclass: str, digits: str, source: str) -> tuple[int, str] | None:
    readings = _readings(digits)
    known = MAIN_GROUPS.get(subclass)
    if known is not None:
        real = [r for r in readings if r[0] in known]
        if real:
            return max(real, key=lambda r: len(str(r[0])))     # longest real main group
        log.warning("%s: no main group of %s fits %r; keeping the subclass only", source, subclass, digits)
        return None
    if len(readings) == 1:
        return readings[0]
    log.warning("%s: subclass %s is not in MAIN_GROUPS and %r can be read %d ways; keeping the subclass only",
                source, subclass, digits, len(readings))
    return None


def split_code(code: str | None) -> Cpc | None:
    """'B60C1112FI' -> Cpc('B60C', 11, '12'). None for an empty or unreadable code."""
    text = re.sub(r"\s+", "", (code or "")).upper()
    if not text:
        return None
    match = _CODE_RE.match(text)
    if not match:
        log.warning("%r is not a CPC code", code)
        return None
    subclass, digits, _suffix = match.groups()
    if not digits:
        return Cpc(subclass)
    parts = _split_digits(subclass, digits, code)
    if parts is None:
        return Cpc(subclass)
    group, subgroup = parts
    return Cpc(subclass, group, subgroup)


def parse_query(text: str) -> Cpc:
    """'B60C', 'B60C 1', 'B60C 11/12' or the stored form 'B60C1112'. '/00' means the whole
    main group. Raises ValueError for anything that is not a usable query."""
    raw = re.sub(r"\s+", " ", (text or "").strip().upper())
    if not raw:
        raise ValueError("Empty classification code.")
    compact = _CODE_RE.match(raw)
    if compact and len(compact.group(2)) >= 1 + MIN_SUBGROUP_DIGITS:
        # stored form without separators, split the same way as the data
        subclass, digits = compact.group(1), compact.group(2)
        parts = _split_digits(subclass, digits, raw)
        if parts is None:
            raise ValueError(f"Cannot tell the main group in {text.strip()!r}; "
                             f"write it as '{subclass} <group>/<subgroup>'.")
        return _checked(subclass, *parts, text)
    match = _QUERY_RE.match(raw)
    if not match:
        raise ValueError(f"{text.strip()!r} is not a CPC code. Use e.g. B60C, B60C 11 or B60C 11/12.")
    subclass, group, subgroup = match.groups()
    if group is None:
        return Cpc(subclass)
    return _checked(subclass, int(group), subgroup, text)


def _checked(subclass: str, group: int, subgroup: str | None, text: str) -> Cpc:
    known = MAIN_GROUPS.get(subclass)
    if known is not None and group not in known:
        raise ValueError(f"{subclass} has no main group {group}. Main groups: "
                         + ", ".join(str(g) for g in sorted(known)) + ".")
    if subgroup is not None and len(subgroup) < MIN_SUBGROUP_DIGITS:
        raise ValueError(f"The subgroup in {text.strip()!r} needs at least {MIN_SUBGROUP_DIGITS} digits "
                         f"(as in {subclass} {group}/00 for the whole group).")
    if subgroup is not None and set(subgroup) == {"0"}:
        subgroup = None                                         # "/00": the whole main group
    return Cpc(subclass, group, subgroup)


def matches(query: Cpc, code: Cpc | None) -> bool:
    """Whole parts must agree. A subgroup also covers the subgroups under it (15/02 covers 15/024)."""
    if code is None or code.subclass != query.subclass:
        return False
    if query.group is None:
        return True
    if code.group != query.group:
        return False
    if query.subgroup is None:
        return True
    return code.subgroup is not None and code.subgroup.startswith(query.subgroup)


def group_title(code: Cpc) -> str | None:
    return GROUP_TITLES.get((code.subclass, code.group)) if code.group is not None else None


CPC_COLUMNS = (("cpc_subclass", "TEXT"), ("cpc_group", "INTEGER"), ("cpc_subgroup", "TEXT"), ("cpc_code", "TEXT"))


def write_cpc_columns(db_path: Path) -> dict:
    """Fill the cpc_* columns of the patents table from classification. Adds the columns to
    an older search.db first, so an existing index can be updated without re-embedding."""
    con = sqlite3.connect(db_path)
    try:
        have = {row[1] for row in con.execute("PRAGMA table_info(patents)")}
        for name, kind in CPC_COLUMNS:
            if name not in have:
                con.execute(f"ALTER TABLE patents ADD COLUMN {name} {kind}")
        rows = con.execute("SELECT doc_number, classification FROM patents").fetchall()
        split = {}
        updates = []
        for doc_number, classification in rows:
            if classification not in split:
                split[classification] = split_code(classification)
            c = split[classification]
            updates.append((
                c.subclass if c else None,
                c.group if c else None,
                (c.subgroup or "00") if c and c.group is not None else None,
                str(c) if c else None,
                doc_number,
            ))
        con.executemany("UPDATE patents SET cpc_subclass = ?, cpc_group = ?, cpc_subgroup = ?, cpc_code = ? "
                        "WHERE doc_number = ?", updates)
        con.commit()
    finally:
        con.close()
    codes = {k: v for k, v in split.items() if k}
    return {"patents": len(rows), "distinct_codes": len(codes),
            "split": sum(1 for v in codes.values() if v and v.group is not None),
            "subclass_only": sum(1 for v in codes.values() if v and v.group is None),
            "unreadable": sum(1 for v in codes.values() if v is None)}


def _subgroup_order(subgroup: str) -> tuple[str, int]:
    """Scheme order: subgroup digits read as a decimal fraction (02 < 024 < 10)."""
    return subgroup.ljust(8, "0"), len(subgroup)


def classification_tree(codes: list[Cpc | None]) -> list[dict]:
    """Subclass -> main group -> subgroup nodes for the codes in the data (one code per
    patent). Each node has `code` (a query string for parse_query), `count` (patents it
    matches, with the same matches() as the filter) and `title` when known."""
    present = [c for c in codes if c is not None]

    def count(query_text: str) -> int:
        q = parse_query(query_text)
        return sum(matches(q, c) for c in present)

    tree = []
    for subclass in sorted({c.subclass for c in present}):
        in_sub = [c for c in present if c.subclass == subclass]
        node = {"code": subclass, "count": count(subclass), "groups": []}
        if subclass in SUBCLASS_TITLES:
            node["title"] = SUBCLASS_TITLES[subclass]
        for group in sorted({c.group for c in in_sub if c.group is not None}):
            g_code = f"{subclass} {group}"
            g_node = {"code": g_code, "count": count(g_code), "subgroups": []}
            if (subclass, group) in GROUP_TITLES:
                g_node["title"] = GROUP_TITLES[(subclass, group)]
            subgroups = {c.subgroup for c in in_sub if c.group == group and c.subgroup and set(c.subgroup) != {"0"}}
            for sub in sorted(subgroups, key=_subgroup_order):
                s_code = f"{subclass} {group}/{sub}"
                g_node["subgroups"].append({"code": s_code, "count": count(s_code)})
            node["groups"].append(g_node)
        tree.append(node)
    return tree
