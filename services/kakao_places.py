import logging
import os

import requests


def get_landmarks_by_keyword(
    lat: float, lng: float, radius_m: int = 1000, query: str = "관광"
) -> list[dict]:
    key = os.environ.get("KAKAO_REST_API_KEY", "")
    if not key:
        return []
    try:
        resp = requests.get(
            "https://dapi.kakao.com/v2/local/search/keyword.json",
            headers={"Authorization": f"KakaoAK {key}"},
            params={
                "query": query,
                "x": lng,
                "y": lat,
                "radius": radius_m,
                "sort": "distance",
            },
            timeout=3,
        )
        resp.raise_for_status()
        places = []
        for p in resp.json().get("documents", []):
            places.append({
                "name": p["place_name"],
                "lat": float(p["y"]),
                "lng": float(p["x"]),
                "address": p.get("road_address_name") or p.get("address_name", ""),
                "distance": int(p.get("distance", 0)),
                "category": p.get("category_name", ""),
            })
        return places
    except Exception:
        logging.warning("Kakao keyword API failed", exc_info=True)
        return []
