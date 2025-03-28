# 전체 과정을 하나의 스크립트로 통합한 코드

import pandas as pd
import json
import os
from ast import literal_eval

# 1. Load and normalize CSVs
def load_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

snack_df = load_csv("raw_data/snack.csv")
item_df = load_csv("raw_data/snack_item.csv")
additive_df = load_csv("raw_data/snack_additive.csv")
map_df = load_csv("raw_data/map_snack_item_additive.csv")

# 2. Rename and map relevant columns
snack_df = snack_df.rename(columns={
    "id": "snack_id",
    "name": "snack_name",
    "snack_type": "category"
})

item_df = item_df.rename(columns={
    "id": "item_id"
})

additive_df = additive_df.rename(columns={
    "id": "additive_id",
    "korean_name": "additive_name",
    "description": "description",
    "grade": "grade",
    "stability_message": "stability"
})

map_df = map_df.rename(columns={
    "snack_item_id": "item_id",
    "snack_additive_id": "additive_id"
})

# 3. Create mappings
additive_info = additive_df.set_index("additive_id")[["additive_name", "grade", "description", "stability"]].to_dict(orient="index")
item_add_map = map_df.groupby("item_id")["additive_id"].apply(list).to_dict()
item_info = item_df.set_index("snack_id").to_dict(orient="index")

# 4. Construct final documents
documents = []

for _, row in snack_df.iterrows():
    snack_id = row["snack_id"]
    item_data = item_info.get(snack_id, {})
    if not item_data:
        continue

    additives = []
    additive_ids = item_add_map.get(item_data.get("item_id"), [])
    additive_names = []

    for aid in additive_ids:
        info = additive_info.get(aid)
        if info:
            additives.append(info)
            additive_names.append(f"{info['additive_name']} ({info['grade']})")

    # Process allergy and safety list
    allergy_list = []
    safe_marks = []
    try:
        if isinstance(row.get("allergy_list"), str):
            allergy_list = literal_eval(row["allergy_list"])
        if isinstance(row.get("safe_food_mark_list"), str):
            marks = literal_eval(row["safe_food_mark_list"])
            safe_marks = [m["markName"] for m in marks if isinstance(m, dict) and "markName" in m]
    except:
        pass

    # Nutrients
    nutrients = {}
    if "nutrient_list" in item_data and isinstance(item_data["nutrient_list"], str):
        try:
            nutrient_list = literal_eval(item_data["nutrient_list"])
            nutrients = {n.get("name", ""): n.get("value", "") for n in nutrient_list if isinstance(n, dict)}
        except:
            pass

    text = f"{row['snack_name']} - {row.get('company', '')}에서 만든 {row.get('category', '')}입니다."
    if allergy_list:
        text += f"\n알러지 정보: {', '.join(allergy_list)}"
    if additive_names:
        text += f"\n첨가물: {', '.join(additive_names)}"

    metadata = {
        "snack_name": row["snack_name"],
        "company": row.get("company", ""),
        "category": row.get("category", ""),
        "thumbnail_url": row.get("thumbnail_url", ""),
        "serving_size": row.get("total_serving_size", ""),
        "calorie": item_data.get("calorie", ""),
        "nutrients": nutrients,
        "allergens": allergy_list,
        "safe_marks": safe_marks,
        "additives": additives,
    }

    documents.append({
        "id": f"snack-{snack_id}",
        "text": text.strip(),
        "metadata": metadata
    })

# Save JSON
output_path = "Snack_data/snack_vector.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(documents, f, ensure_ascii=False, indent=2)

output_path
