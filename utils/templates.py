"""Template-based fallback answers when Gemini is unavailable."""

from __future__ import annotations
import json
from typing import Any, Dict


def fallback_answer(question: str, evidence: Dict[str, Any]) -> str:
    """Produce a deterministic Thai answer from evidence."""
    intent = evidence.get("intent", "general")

    lines = ["## 1) Answer"]

    if intent == "lithology":
        lines.extend(_lithology_answer(evidence))
    elif intent == "flow":
        lines.extend(_flow_answer(evidence))
    elif intent == "hotspot":
        lines.extend(_hotspot_answer(evidence))
    else:
        lines.extend(_general_answer(evidence))

    lines.append("")
    lines.append("## 2) ReasoningPath")
    lines.append(f"- Intent detected: **{intent}**")
    lines.append("- Used GraphRAG in-memory traversal over local GIS data")

    lines.append("")
    lines.append("## 3) DataReferences")
    refs = evidence.get("DataReferences", {})
    for f in refs.get("files", []):
        lines.append(f"- `{f}`")

    return "\n".join(lines)


def _lithology_answer(ev: Dict[str, Any]) -> list:
    ls = ev.get("LithologySummary", {})
    if "note" in ls:
        return [f"- {ls['note']}"]
    lines = [f"- บ่อ **{ls.get('well_id', '?')}** ช่วงความลึก {ls.get('depth_range_filter', 'all')}"]
    for lith in ls.get("lithologies", [])[:5]:
        ranges = ", ".join(lith.get("depth_ranges", [])[:3])
        lines.append(f"  - **{lith['lithology']}**: {lith['thickness_m']} m ({ranges})")
    return lines


def _flow_answer(ev: Dict[str, Any]) -> list:
    dp = ev.get("DownstreamPath", {})
    if "note" in dp:
        return [f"- {dp['note']}"]
    src = dp.get("source_cell", {})
    lines = [f"- บ่ออยู่ใน cell ({src.get('row')},{src.get('col')}) head={src.get('head')}"]
    ds_cells = dp.get("downstream_cells", [])
    lines.append(f"- Downstream cells (≤3 hops): {len(ds_cells)} cells")
    for c in ds_cells[:5]:
        lines.append(f"  - ({c['row']},{c['col']}) head={c['head']}")
    ds_wells = dp.get("downstream_wells", [])
    if ds_wells:
        lines.append(f"- Downstream wells: {', '.join(ds_wells[:10])}")
    else:
        lines.append("- ไม่พบบ่อใน downstream cells")
    return lines


def _hotspot_answer(ev: Dict[str, Any]) -> list:
    lines = []
    hg = ev.get("HotspotGlobal", [])
    if hg:
        lines.append("- **Top lowest-head cells (ทั้งหมด):**")
        for i, c in enumerate(hg[:10], 1):
            wells = ", ".join(c.get("wells_inside", [])) or "ไม่มีบ่อ"
            lines.append(f"  {i}. ({c['row']},{c['col']}) head={c['head']}  → {wells}")
    hn = ev.get("HotspotNear", [])
    if hn:
        lines.append("- **Lowest-head cells ใกล้บ่อที่เลือก (≤5 km):**")
        for c in hn:
            wells = ", ".join(c.get("wells_inside", [])) or "ไม่มีบ่อ"
            lines.append(f"  - ({c['row']},{c['col']}) head={c['head']}  → {wells}")
    return lines


def _general_answer(ev: Dict[str, Any]) -> list:
    lines = []
    sw = ev.get("SelectedWell")
    if sw:
        lines.append(f"- บ่อ **{sw['well_id']}**: depth={sw['total_depth']}m, elev={sw['ground_elev']}m")
    hc = ev.get("HeadCell")
    if hc:
        lines.append(f"- Head cell ({hc['row']},{hc['col']}): head={hc['head']}")
    nw = ev.get("NearbyWells", [])
    if nw:
        lines.append(f"- Nearby wells (≤2 km): {len(nw)}")
    if not lines:
        lines.append("- กรุณาเลือกบ่อและถามคำถามเฉพาะเจาะจงมากขึ้น")
    return lines


def build_report_html(well_summary: Dict, last_answer: str, evidence: Dict) -> str:
    """Build printable 1-page HTML report."""
    html = f"""<!DOCTYPE html>
<html lang="th">
<head><meta charset="utf-8">
<style>
  body {{ font-family: 'Sarabun', sans-serif; max-width: 800px; margin: auto; padding: 20px; background: #ffffff; color: #222222; }}
  h1 {{ color: #1a5276; border-bottom: 2px solid #1a5276; padding-bottom: 8px; }}
  h2 {{ color: #2874a6; }}
  h3 {{ color: #2c3e50; }}
  p, li {{ color: #222222; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; color: #222222; }}
  th {{ background: #eaf2f8; color: #1a5276; }}
  .footer {{ margin-top: 30px; font-size: 0.8em; color: #888; }}
</style></head>
<body>
<h1>Groundwater Intelligence Report</h1>
"""
    # Well summary
    if well_summary:
        html += "<h2>Well Summary</h2><table>"
        for k, v in well_summary.items():
            html += f"<tr><th>{k}</th><td>{v}</td></tr>"
        html += "</table>"

    # Answer
    if last_answer:
        html += f"<h2>Analysis</h2><div>{_md_to_html(last_answer)}</div>"

    # Evidence table
    if evidence:
        html += "<h2>Evidence</h2><table><tr><th>Key</th><th>Value</th></tr>"
        for k, v in evidence.items():
            if k in ("question", "intent", "DataReferences"):
                continue
            val_str = json.dumps(v, ensure_ascii=False, default=str) if isinstance(v, (dict, list)) else str(v)
            if len(val_str) > 300:
                val_str = val_str[:300] + "…"
            html += f"<tr><td>{k}</td><td style='font-size:0.85em'>{val_str}</td></tr>"
        html += "</table>"

    html += "<div class='footer'>Generated by Groundwater Intelligence Copilot (GraphRAG + GeoAI)</div>"
    html += "</body></html>"
    return html


def _md_to_html(md: str) -> str:
    """Very simple markdown-to-html for bullets and headings."""
    lines = md.split("\n")
    out = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            out.append(f"<h3>{stripped[3:]}</h3>")
        elif stripped.startswith("- "):
            out.append(f"<li>{stripped[2:]}</li>")
        elif stripped.startswith("  - "):
            out.append(f"<li style='margin-left:20px'>{stripped[4:]}</li>")
        elif stripped:
            out.append(f"<p>{stripped}</p>")
    return "\n".join(out)
