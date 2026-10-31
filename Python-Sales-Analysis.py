# Python Sales Data Analysis
# Tools: Python, Pandas, Matplotlib
# Dataset: PQ Data.csv

import pandas as pd
import matplotlib.pyplot as plt

# 1. Load data
data = pd.read_csv("PQ Data.csv")

# 2. Explore the dataset
print("Dataset shape:", data.shape)
print("\nColumn information:")
data.info()
print("\nDescriptive statistics:")
print(data.describe(include="all"))

# 3. Data cleaning and type conversion
data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

for column in ["Unit Cost", "Unit Price"]:
    data[column] = pd.to_numeric(
        data[column].astype(str).str.replace(r"[$,]", "", regex=True).str.strip(),
        errors="coerce"
    )

# Check missing values after conversion
print("\nMissing values after conversion:")
print(data.isna().sum())

# Remove rows with missing values in fields required for the analysis
required_columns = [
    "Date", "Customer Age", "Customer Gender", "Country", "State",
    "Product Category", "Sub Category", "Quantity", "Unit Cost", "Unit Price"
]
data = data.dropna(subset=required_columns).copy()

# 4. Feature engineering
data["Total Sales"] = data["Quantity"] * data["Unit Price"]
data["Total Cost"] = data["Quantity"] * data["Unit Cost"]
data["Profit"] = data["Total Sales"] - data["Total Cost"]
data["Profit Margin"] = data["Profit"].div(data["Total Sales"].replace(0, pd.NA))
data["Year"] = data["Date"].dt.year
data["Month"] = data["Date"].dt.month
data["Month Name"] = data["Date"].dt.strftime("%b")
data["Quarter"] = data["Date"].dt.to_period("Q").astype(str)

# 5. Overall KPIs
total_sales = data["Total Sales"].sum()
total_cost = data["Total Cost"].sum()
total_profit = data["Profit"].sum()
total_units = data["Quantity"].sum()
average_unit_price = data["Unit Price"].mean()
overall_profit_margin = total_profit / total_sales if total_sales else 0

print("\nKEY PERFORMANCE INDICATORS")
print(f"Total Sales: {total_sales:,.2f}")
print(f"Total Cost: {total_cost:,.2f}")
print(f"Total Profit: {total_profit:,.2f}")
print(f"Total Units Sold: {total_units:,.0f}")
print(f"Average Unit Price: {average_unit_price:,.2f}")
print(f"Overall Profit Margin: {overall_profit_margin:.2%}")

# 6. Product analysis
category_performance = (
    data.groupby("Product Category")
    .agg(
        Total_Sales=("Total Sales", "sum"),
        Total_Profit=("Profit", "sum"),
        Units_Sold=("Quantity", "sum")
    )
    .sort_values("Total_Sales", ascending=False)
)
print("\nSales and profit by product category:")
print(category_performance)

subcategory_performance = (
    data.groupby("Sub Category")
    .agg(Total_Sales=("Total Sales", "sum"), Total_Profit=("Profit", "sum"))
    .sort_values("Total_Sales", ascending=False)
)
print("\nTop 5 sub-categories by sales:")
print(subcategory_performance.head(5))

# 7. Geographic analysis
country_performance = (
    data.groupby("Country")
    .agg(Total_Sales=("Total Sales", "sum"), Total_Profit=("Profit", "sum"))
    .sort_values("Total_Sales", ascending=False)
)
print("\nSales by country:")
print(country_performance)

top_states = (
    data.groupby("State")["Total Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)
print("\nTop 5 states by sales:")
print(top_states)

# 8. Customer analysis
age_summary = data["Customer Age"].describe()
gender_distribution = data["Customer Gender"].value_counts()
print("\nCustomer age summary:")
print(age_summary)
print("\nRecords by customer gender:")
print(gender_distribution)

# 9. Time-series analysis
monthly_sales = (
    data.set_index("Date")
    .resample("MS")["Total Sales"]
    .sum()
)
yearly_sales = data.groupby("Year")["Total Sales"].sum()
print("\nSales by year:")
print(yearly_sales)
print("\nTop 5 months by sales:")
print(monthly_sales.sort_values(ascending=False).head(5))

# 10. Visualizations
category_performance["Total_Sales"].sort_values().plot(
    kind="barh", title="Total Sales by Product Category", figsize=(8, 5)
)
plt.xlabel("Total Sales")
plt.ylabel("Product Category")
plt.tight_layout()
plt.show()

monthly_sales.plot(
    figsize=(11, 5), marker="o", title="Monthly Sales Trend"
)
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.show()

gender_distribution.plot(
    kind="bar", title="Records by Customer Gender", figsize=(6, 4)
)
plt.xlabel("Customer Gender")
plt.ylabel("Number of Records")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 11. Export the cleaned and enriched dataset
data.to_csv("cleaned_sales_analysis.csv", index=False)
print("\nCleaned dataset exported to cleaned_sales_analysis.csv")
