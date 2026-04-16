import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import os
import re

EUR_TO_USD = 1.2

def clean_price(value):
    if pd.isna(value):
        return 0.0
    cleaned = re.sub(r"[^0-9\.\,]", "", str(value))
    cleaned = cleaned.replace(",", ".")
    try:
        return float(cleaned)
    except:
        return 0.0

def process_folder(folder):
    users, orders, books = None, None, None
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)

        if fname.lower().startswith("users") and fname.endswith(".csv"):
            users = pd.read_csv(path)
            users = users.dropna(subset=["id","name","address","phone","email"])
            users["id"] = pd.to_numeric(users["id"], errors="coerce")
            users = users.dropna(subset=["id"])
            users["id"] = users["id"].astype(int)

        elif fname.lower().startswith("orders") and fname.endswith(".parquet"):
            orders = pd.read_parquet(path)
            orders["quantity"] = pd.to_numeric(orders["quantity"], errors="coerce").fillna(1).astype(int)
            orders["unit_price"] = orders["unit_price"].apply(clean_price)

            orders["timestamp"] = pd.to_datetime(orders["timestamp"], errors="coerce", utc=True)
            orders = orders.dropna(subset=["timestamp"])

            orders = orders[orders["timestamp"].dt.year <= 2025]
            orders = orders[(orders["unit_price"] > 0) & (orders["unit_price"] < 10000)]

            orders["paid_price"] = orders["quantity"] * orders["unit_price"] * EUR_TO_USD
            orders["date"] = orders["timestamp"].dt.date

        elif fname.lower().startswith("books") and fname.endswith(".yaml"):
            with open(path, "r") as f:
                books_data = yaml.safe_load(f)
            normalized = []
            for entry in books_data:
                fixed = {}
                for k, v in entry.items():
                    key = k.lstrip(":")
                    if key == "author":
                        key = "authors"
                    fixed[key] = v
                normalized.append(fixed)
            books = pd.DataFrame(normalized)
            books = books.dropna(subset=["id","authors"])
            books["authors"] = books["authors"].apply(
                lambda x: [a.strip() for a in str(x).split(",") if a.strip()]
            )
    return users, orders, books

def analyze_folder(folder):
    users, orders, books = process_folder(folder)
    if users is None or orders is None or books is None:
        return None

    daily_revenue = orders.groupby("date")["paid_price"].sum()
    top_days = daily_revenue.sort_values(ascending=False).head(5)

    unique_users = users["id"].nunique()

    all_authors = []
    for auth_list in books["authors"].dropna():
        all_authors.extend(auth_list)
    author_counts = pd.Series(all_authors).value_counts()
    unique_author_sets = len(author_counts)
    max_count = author_counts.max() if not author_counts.empty else 0
    popular_authors = author_counts[author_counts == max_count].index.tolist()

    buyer_spending = orders.groupby("user_id")["paid_price"].sum()
    best_buyer = pd.DataFrame()
    alias_ids = []
    if not buyer_spending.empty:
        best_buyer_id = buyer_spending.idxmax()
        best_buyer = users.loc[users["id"] == best_buyer_id]
        if not best_buyer.empty:
            best_email = best_buyer["email"].iloc[0]
            alias_ids = users.loc[users["email"] == best_email, "id"].tolist()

    return {
        "top_days": top_days,
        "unique_users": unique_users,
        "unique_author_sets": unique_author_sets,
        "popular_authors": popular_authors,
        "best_buyer": best_buyer,
        "alias_ids": alias_ids,
        "daily_revenue": daily_revenue
    }

st.title("📊 Task 4 Dashboard")
st.markdown("This dashboard analyzes user, order, and book data across three datasets (DATA1, DATA2, DATA3).")

for folder in ["DATA1","DATA2","DATA3"]:
    st.header(folder)
    results = analyze_folder(folder)
    if results is None:
        st.write("Missing files in", folder)
        continue

    st.write("**Top 5 revenue days:**")
    if not results["top_days"].empty:
        st.table(
            results["top_days"]
            .reset_index()
            .rename(columns={"date": "Date", "paid_price": "Revenue (USD)"})
            .style.format({"Revenue (USD)": "{:.2f}"})
        )
    else:
        st.write("No revenue data available")

    st.write("**Unique users:**", results["unique_users"])
    st.write("**Unique authors (counted individually):**", results["unique_author_sets"])
    st.write("**Most popular author(s):**", results["popular_authors"] if results["popular_authors"] else "None")
    st.write("**Best buyer aliases:**", results["alias_ids"] if results["alias_ids"] else "None")

    fig, ax = plt.subplots()
    if not results["daily_revenue"].empty:
        results["daily_revenue"].plot(kind="line", marker="o", ax=ax)
    ax.set_title(f"Daily Revenue - {folder}")
    ax.set_ylabel("Revenue (USD)")
    st.pyplot(fig)
