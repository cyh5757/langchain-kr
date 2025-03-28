import json
import pandas as pd

# 하드코딩된 경로 (Windows 환경)
base_path = r"C:\Users\MAIN\Workplace\LLM\langchain-kr\langchain-kr\raw_data"

# 파일 경로 설정
snack_path = f"{base_path}\\snack.csv"
item_path = f"{base_path}\\snack_item.csv"
additive_path = f"{base_path}\\snack_additive.csv"
map_path = f"{base_path}\\map_snack_item_additive.csv"

# 데이터 로드
snack_df = pd.read_csv(snack_path)
item_df = pd.read_csv(item_path)
additive_df = pd.read_csv(additive_path)
map_df = pd.read_csv(map_path)

# 매핑 준비
snack_info = snack_df.set_index("id")[["name", "company", "barcode", "total_serving_size"]].to_dict("index")
additive_info = additive_df.set_index("id")[[
    "korean_name", "english_name", "main_use_list", "grade", "description", "stability_message"
]].to_dict("index")
item_additives = map_df.groupby("snack_item_id")["snack_additive_id"].apply(list)

# JSON 필드 처리
item_df["nutrient_list"] = item_df["nutrient_list"].apply(lambda x: json.loads(x) if isinstance(x, str) else [])
item_df["serving_size"] = item_df["serving_size"].apply(lambda x: json.loads(x) if isinstance(x, str) else {"amount": None, "amountUnit": ""})

# 문서 생성
documents = []
for _, row in item_df.iterrows():
    snack_id = row["snack_id"]
    snack_meta = snack_info.get(snack_id, {})
    additives = []

    for aid in item_additives.get(row["id"], []):
        ainfo = additive_info.get(aid)
        if ainfo:
            additives.append({
                "name": ainfo["korean_name"],
                "english_name": ainfo["english_name"],
                "grade": ainfo["grade"],
                "main_use": json.loads(ainfo["main_use_list"]) if isinstance(ainfo["main_use_list"], str) else [],
                "description": ainfo["description"],
                "stability": ainfo["stability_message"]
            })

    serving = row["serving_size"]
    serving_str = f"{serving.get('amount', '')}{serving.get('amountUnit', '')}"

    page_content = (
        f"Snack: {snack_meta.get('name', 'N/A')}\n"
        f"Company: {snack_meta.get('company', 'N/A')}\n"
        f"Barcode: {snack_meta.get('barcode', 'N/A')}\n"
        f"Total Serving Size: {snack_meta.get('total_serving_size', 'N/A')}\n"
        f"Item: {row.get('item_name', 'N/A')}, Calorie: {row.get('calorie', 'N/A')}, Serving Size: {serving_str}\n"
        f"Nutrients: {json.dumps(row.get('nutrient_list', []), ensure_ascii=False)}\n"
        f"Additives: {', '.join([a['name'] for a in additives]) if additives else 'None'}"
    )

    metadata = {
        "snack_name": snack_meta.get("name"),
        "company": snack_meta.get("company"),
        "barcode": snack_meta.get("barcode"),
        "total_serving_size": snack_meta.get("total_serving_size"),
        "item_name": row.get("item_name"),
        "calorie": row.get("calorie"),
        "serving_size": serving_str,
        "nutrients": row.get("nutrient_list"),
        "additives": additives
    }

    documents.append({
        "page_content": page_content,
        "metadata": metadata
    })

# 저장 경로
output_path = r"C:\Users\MAIN\Workplace\LLM\langchain-kr\langchain-kr\Snack_data\snack_data.json"

# JSON 파일 저장
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(documents, f, ensure_ascii=False, indent=4)

output_path