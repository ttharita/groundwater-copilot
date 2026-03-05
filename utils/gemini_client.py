"""OpenAI LLM client – reads API key from Streamlit secrets or env var."""

from __future__ import annotations
import json
import os
from typing import Any, Dict, Optional

import requests
import streamlit as st


OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"


def _get_api_key() -> Optional[str]:
    """Return OpenAI API key or None."""
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("OPENAI_API_KEY")


def is_available() -> bool:
    key = _get_api_key()
    return bool(key and key.strip())


def generate_answer(question: str, evidence: Dict[str, Any]) -> str:
    """Call OpenAI and return the Thai answer string."""
    api_key = _get_api_key()
    if not api_key:
        return "[OpenAI API key not configured – showing template answer only]"

    prompt = _build_prompt(question, evidence)

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a hydrogeology expert. Always respond in Thai."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 700,
    }

    try:
        resp = requests.post(
            OPENAI_ENDPOINT,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return "[OpenAI returned no content]"
    except requests.exceptions.HTTPError as e:
        body = ""
        try:
            body = resp.text[:300]
        except Exception:
            pass
        return f"[OpenAI API error: {e}]\n{body}"
    except Exception as e:
        return f"[OpenAI call failed: {e}]"


def _build_prompt(question: str, evidence: Dict[str, Any]) -> str:
    evidence_json = json.dumps(evidence, ensure_ascii=False, indent=2, default=str)
    return f"""คุณคือผู้เชี่ยวชาญด้านอุทกธรณีวิทยา (Hydrogeology Expert) สำหรับผู้บริหาร

คำถาม: {question}

หลักฐาน (Evidence JSON):
{evidence_json}

กฎ:
- ตอบเป็นภาษาไทย สั้น กระชับ เหมาะผู้บริหาร
- ใช้ bullet points
- ใช้ข้อมูลจาก Evidence JSON เท่านั้น ห้ามคิดตัวเลขเอง
- หากข้อมูลไม่เพียงพอ ให้ระบุว่าขาดอะไร และแนะนำข้อมูลที่ควรเพิ่ม
- ตอบตามรูปแบบด้านล่างเท่านั้น

รูปแบบ:
## 1) Answer
(คำตอบหลัก)

## 2) ReasoningPath
(อธิบายเหตุผลสั้นๆ)

## 3) DataReferences
(ระบุแหล่งข้อมูลที่ใช้)
"""
