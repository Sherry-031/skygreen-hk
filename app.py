"""SkyGreen HK - low-altitude urban planning sandbox.

Run with:
    streamlit run app.py

Expected files:
    data/routes.csv
    data/routes.geojson
    data/risk_zones.geojson
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import folium
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium


st.set_page_config(
    page_title="SkyGreen HK",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ROUTES_CSV = DATA_DIR / "routes.csv"
ROUTES_GEOJSON = DATA_DIR / "routes.geojson"
RISK_GEOJSON = DATA_DIR / "risk_zones.geojson"

ROUTE_COLORS = {
    "ground route": "#737373",
    "fastest drone route": "#D73027",
    "green corridor": "#1A9850",
    "ai green corridor": "#1A9850",
}

RISK_COLORS = {
    "population": "#F59E0B",
    "dense population": "#F59E0B",
    "sensitive facility": "#2563EB",
    "restricted": "#DC2626",
    "regulatory": "#DC2626",
    "preferred corridor": "#22C55E",
}


def normalise(text: Any) -> str:
    return str(text or "").strip().lower().replace("_", " ")


def first_existing(columns: list[str], aliases: list[str]) -> str | None:
    lookup = {normalise(column): column for column in columns}
    for alias in aliases:
        if normalise(alias) in lookup:
            return lookup[normalise(alias)]
    return None


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_geojson(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def property_value(feature: dict[str, Any], aliases: list[str], default: str) -> str:
    properties = feature.get("properties") or {}
    lookup = {normalise(key): value for key, value in properties.items()}
    for alias in aliases:
        if normalise(alias) in lookup:
            return str(lookup[normalise(alias)])
    return default


def route_name(feature: dict[str, Any]) -> str:
    return property_value(
        feature,
        ["route_name", "route name", "name", "route", "scenario"],
        "Route",
    )


def route_color(name: str) -> str:
    key = normalise(name)
    if key in ROUTE_COLORS:
        return ROUTE_COLORS[key]
    if "ground" in key:
        return ROUTE_COLORS["ground route"]
    if "fast" in key or "direct" in key:
        return ROUTE_COLORS["fastest drone route"]
    if "green" in key or "sustainable" in key:
        return ROUTE_COLORS["green corridor"]
    return "#6B7280"


def risk_type(feature: dict[str, Any]) -> str:
    return property_value(
        feature,
        ["zone_type", "zone type", "type", "category"],
        "Risk zone",
    )


def risk_color(name: str) -> str:
    key = normalise(name)
    if key in RISK_COLORS:
        return RISK_COLORS[key]
    if "population" in key or "dense" in key:
        return "#F59E0B"
    if "hospital" in key or "school" in key or "sensitive" in key:
        return "#2563EB"
    if "restrict" in key or "regulat" in key:
        return "#DC2626"
    if "preferred" in key or "corridor" in key:
        return "#22C55E"
    return "#7C3AED"


def all_coordinates(geometry: dict[str, Any] | None) -> list[list[float]]:
    if not geometry:
        return []
    coordinates = geometry.get("coordinates", [])
    geometry_type = geometry.get("type", "")
    if geometry_type == "Point":
        return [coordinates]
    if geometry_type in {"LineString", "MultiPoint"}:
        return coordinates
    if geometry_type == "Polygon":
        return [point for ring in coordinates for point in ring]
    if geometry_type == "MultiLineString":
        return [point for line in coordinates for point in line]
    if geometry_type == "MultiPolygon":
        return [point for polygon in coordinates for ring in polygon for point in ring]
    return []


def build_map(routes: dict[str, Any], risks: dict[str, Any]) -> folium.Map:
    coordinates: list[list[float]] = []
    for collection in (routes, risks):
        for feature in collection.get("features", []):
            coordinates.extend(all_coordinates(feature.get("geometry")))

    if coordinates:
        mean_lon = sum(point[0] for point in coordinates) / len(coordinates)
        mean_lat = sum(point[1] for point in coordinates) / len(coordinates)
        centre = [mean_lat, mean_lon]
    else:
        centre = [22.3027, 114.1772]

    map_object = folium.Map(
        location=centre,
        zoom_start=12,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    risks_group = folium.FeatureGroup(name="Risk and planning zones", show=True)
    for feature in risks.get("features", []):
        zone_type = risk_type(feature)
        colour = risk_color(zone_type)
        zone_name = property_value(feature, ["zone_name", "name"], zone_type)
        risk_level = property_value(feature, ["risk_level", "risk level"], "N/A")
        folium.GeoJson(
            feature,
            style_function=lambda _feature, c=colour: {
                "fillColor": c,
                "color": c,
                "weight": 1.5,
                "fillOpacity": 0.30,
            },
            tooltip=folium.Tooltip(
                f"{zone_name}<br>Type: {zone_type}<br>Risk level: {risk_level}"
            ),
        ).add_to(risks_group)
    risks_group.add_to(map_object)

    for feature in routes.get("features", []):
        name = route_name(feature)
        colour = route_color(name)
        group = folium.FeatureGroup(name=name, show=True)
        folium.GeoJson(
            feature,
            style_function=lambda _feature, c=colour: {
                "color": c,
                "weight": 6,
                "opacity": 0.90,
            },
            tooltip=folium.Tooltip(name),
        ).add_to(group)
        group.add_to(map_object)

    if coordinates:
        lats = [point[1] for point in coordinates]
        lons = [point[0] for point in coordinates]
        map_object.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

    folium.LayerControl(collapsed=False).add_to(map_object)
    return map_object


def prepare_metrics(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    columns = list(raw.columns)
    aliases = {
        "方案": ["route_name", "route name", "name", "scenario", "方案"],
        "时间（分钟）": ["time_min", "time", "duration_min", "时间"],
        "距离（km）": ["distance_km", "distance", "距离"],
        "能耗（kWh）": ["energy_kwh", "energy", "能耗"],
        "碳排放（kgCO₂e）": ["carbon_kg", "carbon", "emissions", "co2", "碳排放"],
        "噪声风险": ["noise_score", "noise", "noise risk", "噪声"],
        "人口暴露风险": [
            "population_score",
            "population exposure",
            "population risk",
            "人口暴露",
        ],
        "监管风险": ["regulatory_score", "regulatory risk", "regulation", "监管风险"],
    }
    selected: dict[str, str] = {}
    for display_name, candidates in aliases.items():
        match = first_existing(columns, candidates)
        if match:
            selected[display_name] = match

    result = raw[[column for column in selected.values()]].copy()
    result = result.rename(columns={value: key for key, value in selected.items()})
    return result, selected


def risk_label(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    labels = {1: "低", 2: "较低", 3: "中", 4: "较高", 5: "高"}
    return labels.get(round(number), f"{number:g}")


st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 1.5rem;}
    .skygreen-subtitle {color: #48665a; margin-top: -0.7rem;}
    .disclaimer {padding: 0.9rem; border-left: 4px solid #1A9850;
                 background: #eef8f1; color: #294437;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌿 SkyGreen HK")
st.markdown(
    '<p class="skygreen-subtitle">AI-powered Low-Altitude Urban Planning Sandbox</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Scenario settings")
    origin = st.text_input("起点 / Origin", "The University of Hong Kong")
    destination = st.text_input("终点 / Destination", "West Kowloon")
    cargo_type = st.selectbox(
        "货物类型 / Cargo type",
        ["Emergency medical supplies", "General parcel", "Food and essentials"],
    )
    weight = st.number_input(
        "重量 / Weight (kg)", min_value=0.1, max_value=25.0, value=5.0, step=0.5
    )
    preference = st.selectbox(
        "决策偏好 / Decision preference",
        ["Balanced", "Fastest", "Lowest carbon", "Lowest social impact"],
    )
    st.divider()
    st.caption(f"{origin} → {destination}")
    st.caption(f"{cargo_type} · {weight:g} kg · {preference}")

required_paths = [ROUTES_CSV, ROUTES_GEOJSON, RISK_GEOJSON]
missing_paths = [path for path in required_paths if not path.exists()]
if missing_paths:
    st.error("缺少以下数据文件，请确认它们位于app.py旁边的data文件夹中：")
    for path in missing_paths:
        st.code(str(path.relative_to(BASE_DIR)))
    st.stop()

try:
    routes_df_raw = load_csv(ROUTES_CSV)
    routes_geojson = load_geojson(ROUTES_GEOJSON)
    risks_geojson = load_geojson(RISK_GEOJSON)
except Exception as error:
    st.error(f"读取数据失败：{error}")
    st.stop()

metrics_df, detected_columns = prepare_metrics(routes_df_raw)
if "方案" not in metrics_df.columns:
    st.error(
        "routes.csv中没有找到方案名称列。请将该列命名为route_name、name或scenario。"
    )
    st.stop()

map_column, metrics_column = st.columns([1.65, 1], gap="large")

with map_column:
    st.subheader("Scenario map")
    scenario_map = build_map(routes_geojson, risks_geojson)
    st_folium(scenario_map, height=610, use_container_width=True, returned_objects=[])
    st.caption("灰色：地面运输　红色：最快无人机路线　绿色：可持续低空走廊")

with metrics_column:
    st.subheader("Scenario comparison")
    display_df = metrics_df.copy()
    for column in ["噪声风险", "人口暴露风险", "监管风险"]:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(risk_label)
    st.dataframe(display_df.set_index("方案"), use_container_width=True, height=300)

    st.markdown("#### Recommendation")
    recommended = {
        "Balanced": "Green Corridor",
        "Fastest": "Fastest Drone Route",
        "Lowest carbon": "Green Corridor",
        "Lowest social impact": "Green Corridor",
    }[preference]
    st.success(f"Recommended scenario: {recommended}")
    st.caption(
        "The recommendation is illustrative in this prototype and should be "
        "recalculated when validated operational data become available."
    )

st.subheader("Indicator comparison")
numeric_metrics = [
    column
    for column in metrics_df.columns
    if column != "方案" and pd.api.types.is_numeric_dtype(metrics_df[column])
]
if numeric_metrics:
    selected_metric = st.selectbox("选择比较指标", numeric_metrics, index=0)
    chart = px.bar(
        metrics_df,
        x="方案",
        y=selected_metric,
        color="方案",
        color_discrete_map={
            name: route_color(name) for name in metrics_df["方案"].astype(str).tolist()
        },
        text_auto=".3g",
    )
    chart.update_layout(
        showlegend=False,
        xaxis_title=None,
        yaxis_title=selected_metric,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(chart, use_container_width=True)
else:
    st.warning("routes.csv中没有可用于绘图的数值指标列。")

st.markdown(
    """
    <div class="disclaimer">
    <strong>Important:</strong> This prototype is a planning sandbox and does not
    provide operational flight navigation or regulatory approval.
    </div>
    """,
    unsafe_allow_html=True,
)
