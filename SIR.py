import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

import os
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'


st.set_page_config(page_title="Interactive Inventory & Reorder Dashboard", layout="wide")

# === Base Data ===
data = {
    'SKU': ['40x40', '45x45', '45x70', '50x50', '55x55', '60x60', '65x65', '70x70', '35x55'],
    'Average On Hand': [36, 39, 35, 144, 110, 146, 103, 31, 52],
    'Average Transfer': [12, 17, 10, 37, 36, 53, 35, 7, 19],
    'Average On Hand Cost': [3330.4, 3971.2, 7525, 16454, 14520, 23581.13,20354.4, 7567.4, 4811.2],
    'COGS': [16560, 26010, 32250, 63270, 71280, 127995.00, 103950, 25305, 26505],
    'SOGS': [30600,47175, 57750.00,116550.00, 129600.00, 234525.00, 189000.00, 45150.00, 49875.00],
    'T/O': [5, 6.5, 4.3, 3.8, 4.9, 5.4, 5.1, 3.3, 5.5],
    'DOH': [63, 48, 73, 81, 64, 57, 61, 93, 57],
    'Sell Through Rate': [85.31 , 87.77, 100, 94.69, 95.62, 100, 95.53, 92.13, 95.99],
    'GMROI': [2.3, 2.9, 1.9, 1.8, 2.2, 2.5, 2.3, 1.5, 2.6],
    'Current Price (CP)': [92, 102, 215, 114, 132, 161, 198, 241, 93]  # <-- Fill these in manually
}


df_base = pd.DataFrame(data)
cost = pd.read_excel(r"C:\Users\InventoryPC-1\Desktop\cost_feathers_with_size.xlsx", sheet_name='Sheet1')
cost_tot= pd.read_excel(r"C:\Users\InventoryPC-1\Downloads\formatted_purchase_data.xlsx")
months = [
    'Oct-20', 'Nov-20', 'Dec-20', 'Jan-21', 'Feb-21', 'Mar-21', 'Apr-21', 'May-21', 'Jun-21', 'Jul-21', 'Aug-21', 'Sep-21',
    'Oct-21', 'Nov-21', 'Dec-21', 'Jan-22', 'Feb-22', 'Mar-22', 'Apr-22', 'May-22', 'Jun-22', 'Jul-22', 'Aug-22',
    'Sep-22', 'Oct-22', 'Nov-22', 'Dec-22', 'Jan-23', 'Feb-23', 'Mar-23', 'Apr-23', 'May-23', 'Jun-23', 'Jul-23', 'Aug-23',
    'Sep-23', 'Oct-23', 'Nov-23', 'Dec-23', 'Jan-24', 'Feb-24', 'Mar-24', 'Apr-24', 'May-24', 'Jun-24', 'Jul-24', 'Aug-24',
    'Sep-24', 'Oct-24', 'Nov-24', 'Dec-24', 'Jan-25', 'Feb-25', 'Mar-25', 'Apr-25', 'May-25', 'Jun-25', 'Jul-25', 'Aug-25', 'Sep-25'
]

sales_raw = """97646.17 85127.15 84954.20 70862.58 53240.29 66482.56 64858.94 65081.63 64143.31 75281.47 78498.06 63087.33
90726.29 104655.12 85947.56 59960.48 59691.05 74721.79 59379.41 67341.50 57017.49 60687.31 63460.91
66054.00 70509.65 94843.26 79788.53 72088.39 60593.40 61045.67 56785.29 56249.98 60789.11 77650.55 81432.78 
98242.79 80534.86 117785.91 94240.20 61676.55 34187.99 75616.24 88364.80 68400.89 62709.15 91110.70 67244.83 
85896.38 98058.85 111796.26 95626.00 74298.54 59764.74 61748.87 69040.78 75670.26 66750.48 58919.03 61827.98 80000"""


sales_values = [float(s) for s in sales_raw.split()]
sales_df = pd.DataFrame({'Month': months, 'Sales': sales_values})
avg_sales = sales_df['Sales'].mean()
sales_df['Multiplier'] = sales_df['Sales'] / avg_sales

today = datetime.today().date()
forecast_months = pd.date_range(start=(today + pd.offsets.MonthBegin(1)), periods=5, freq='MS').strftime('%b-%y').tolist()

# Sidebar with grouped inputs using expanders for better organization

with st.sidebar:
    st.header("Settings & Parameters")

    with st.expander("Global Cost & Time Parameters", expanded=True):
        ordering_cost = st.number_input("Ordering Cost per Order (ZAR)", 100, 500, 250, 10)
        holding_cost_rate = st.slider("Annual Holding Cost Rate (%)", 0.1, 1.0, 0.30, 0.01)
        lead_time_days = st.slider("Lead Time (days)", 7, 30, 14)
        reorder_buffer_days = st.slider("Reorder Buffer (days before lead time)", 1, 7, 14)

    with st.expander("Goals", expanded=True):
        goal_to = st.number_input("Goal: Turnover (T/O)", 1.0, 20.0, 6.0, 0.1)
        goal_doh = st.number_input("Goal: Days on Hand (DOH)", 10, 120, 45, 1)
        goal_gmroi = st.number_input("Goal: GMROI (x)", 0.1, 10.0, 2.5, 0.1)
        goal_str = st.number_input("Goal: Sell Through Rate (%)", 50, 150, 80, 1)

    with st.expander("SKU Selection", expanded=True):
        selected_skus = st.multiselect("Select SKUs", df_base['SKU'].tolist(), default=df_base['SKU'].tolist())

    with st.expander("Adjust SKU Stock & Demand", expanded=True):
        df_filtered = df_base[df_base['SKU'].isin(selected_skus)].copy()
        df_filtered.reset_index(drop=True, inplace=True)

        for idx, row in df_filtered.iterrows():
            current_stock = st.number_input(f"Current Stock - {row['SKU']}", min_value=0, max_value=1000, value=int(row['Average On Hand']), key=f"stock_{row['SKU']}")
            avg_transfer = st.number_input(f"Avg Monthly Demand - {row['SKU']}", min_value=0.0, max_value=500.0, value=float(row['Average Transfer']), step=0.1, key=f"demand_{row['SKU']}")
            df_filtered.loc[idx, 'Current on Hand'] = current_stock
            df_filtered.loc[idx, 'Average Transfer'] = avg_transfer

# Calculate Unit Cost, Holding Cost, EOQ
df_filtered['Unit Cost'] = df_filtered['Average On Hand Cost'] / df_filtered['Average On Hand']
df_filtered['Holding Cost'] = df_filtered['Unit Cost'] * holding_cost_rate
df_filtered['Annual Demand'] = df_filtered['Average Transfer'] * 12
df_filtered['EOQ'] = np.sqrt((2 * df_filtered['Annual Demand'] * ordering_cost) / df_filtered['Holding Cost'])
df_filtered['EOQ (units)'] = df_filtered['EOQ'].round().astype(int)
df_filtered['Reorder Point (units)'] = (df_filtered['Average Transfer'] * (lead_time_days / 30)).round().astype(int)

# Build monthly demand with multipliers for simulation
# --- Replace this block (build monthly demand with multipliers) ---

# Build monthly demand with multipliers for simulation
monthly_demands = pd.DataFrame(index=df_filtered['SKU'], columns=forecast_months, dtype=float)
multiplier_lookup = dict(zip(sales_df['Month'], sales_df['Multiplier']))

for sku in df_filtered['SKU']:
    base_monthly_demand = df_filtered.loc[df_filtered['SKU'] == sku, 'Average Transfer'].values[0]
    monthly_demands.loc[sku] = [base_monthly_demand * multiplier_lookup.get(month, 1.0) for month in forecast_months]


# Simulation: inventory levels and reorder points with adjusted lead time
sim_inventory = []
sim_reorders = []

for sku in df_filtered['SKU']:
    stock = df_filtered.loc[df_filtered['SKU'] == sku, 'Current on Hand'].values[0]
    eoq = df_filtered.loc[df_filtered['SKU'] == sku, 'EOQ (units)'].values[0]
    avg_demand = df_filtered.loc[df_filtered['SKU'] == sku, 'Average Transfer'].values[0]
    min_stock_threshold = avg_demand * ((lead_time_days + reorder_buffer_days) / 30)  # Reorder trigger threshold in days

    for month in forecast_months:
        demand = avg_demand * multiplier_lookup.get(month, 1.0)

        # Reorder when stock is below threshold
        if stock <= min_stock_threshold:
            stock += eoq
            reorder_date = datetime.strptime(month, '%b-%y') + timedelta(days=lead_time_days)  # Adjust to 1-2 weeks lead time
            sim_reorders.append({'SKU': sku, 'Month': month, 'Reorder Qty': eoq, 'Reorder Date': reorder_date.strftime('%Y-%m-%d')})

        stock -= demand
        stock = max(stock, 0)
        sim_inventory.append({'SKU': sku, 'Month': month, 'On Hand': stock})

# Continue with DataFrame creation for simulation
sim_inv_df = pd.DataFrame(sim_inventory)
sim_reorder_df = pd.DataFrame(sim_reorders)

# Reorder Advice for next month
next_month = forecast_months[0]

advice = []

for sku in df_filtered['SKU']:
    inv_row = sim_inv_df[(sim_inv_df['SKU'] == sku) & (sim_inv_df['Month'] == next_month)]

    if not inv_row.empty:
        current_stock = inv_row['On Hand'].values[0]
        reorder_row = sim_reorder_df[(sim_reorder_df['SKU'] == sku) & (sim_reorder_df['Month'] == next_month)]
        reorder_qty = reorder_row['Reorder Qty'].values[0] if not reorder_row.empty else 0

    else:
        current_stock = df_filtered.loc[df_filtered['SKU'] == sku, 'Current on Hand'].values[0]
        reorder_qty = 0

    demand_next_month = monthly_demands.loc[sku, next_month]
    daily_demand = demand_next_month / 30 if demand_next_month > 0 else 0
    days_left = current_stock / daily_demand if daily_demand > 0 else float('inf')

    days_to_reorder = days_left - lead_time_days - reorder_buffer_days

    if days_to_reorder <= 0:
        reorder_status = 'Reorder NOW'
        reorder_date = today.strftime('%Y-%m-%d')
    else:
        reorder_status = f'Reorder in {int(days_to_reorder)} days'
        reorder_date = (today + timedelta(days=days_to_reorder)).strftime('%Y-%m-%d')

    advice.append({
        'SKU': sku,
        'Days of Inventory Left': round(days_left, 1),
        'Reorder Status': reorder_status,
        'Suggested Reorder Date': reorder_date,
        'Reorder Qty from Sim': reorder_qty
    })

reorder_advice_df = pd.DataFrame(advice)


# KPI Calculations on filtered SKUs
avg_to = df_filtered['T/O'].mean()
avg_doh = df_filtered['DOH'].mean()
avg_gmroi = df_filtered['GMROI'].mean()
avg_str = df_filtered['Sell Through Rate'].mean()

def format_metric(value, goal, higher_is_better=True):
    # Calculate delta
    if higher_is_better:
        delta = value - goal
        delta_str = f"{delta:+.2f}"
    else:
        delta = goal - value  # invert for lower is better
        delta_str = f"{delta:+.2f}"

    # Decide delta color icon
    if delta >= 0:
        delta_display = f"🟢 {delta_str}"
    else:
        delta_display = f"🔴 {delta_str}"

    return delta_display

st.title("📦 Interactive Inventory & Reorder Dashboard (ZAR)")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    label="Avg Turnover (T/O)",
    value=f"{avg_to:.2f}",
    delta=format_metric(avg_to, goal_to, higher_is_better=True)
)

col2.metric(
    label="Avg Days On Hand (DOH)",
    value=f"{avg_doh:.1f} days",
    delta=format_metric(avg_doh, goal_doh, higher_is_better=False)  # lower is better here
)

col3.metric(
    label="Avg GMROI",
    value=f"{avg_gmroi:.2f}x",
    delta=format_metric(avg_gmroi, goal_gmroi, higher_is_better=True)
)

col4.metric(
    label="Avg Sell-Thru Rate",
    value=f"{avg_str:.1f}%",
    delta=format_metric(avg_str, goal_str, higher_is_better=True)
)


st.markdown("---")

# Inventory and EOQ table (editable stock and demand reflected)
st.markdown("### SKU Inventory Details feathers")
styled_df = df_filtered.style.format({
    'Average On Hand': '{:.0f}',
    'Current on Hand': '{:.0f}',
    'Average Transfer': '{:.1f}',
    'Average On Hand Cost': 'R{:,.2f}',
    'COGS': 'R{:,.2f}',
    'SOGS': 'R{:,.2f}',
    'EOQ (units)': '{:.0f}',
    'Reorder Point (units)': '{:.0f}',
    'Unit Cost': 'R{:,.2f}',
    'Holding Cost': 'R{:,.2f}',
    'Annual Demand': '{:.0f}',
    'T/O': '{:.2f}',
    'DOH': '{:.0f}',
    'Sell Through Rate': '{:.1f}%',
    'GMROI': '{:.2f}x'
}).applymap(lambda v: 'color: green; font-weight: bold' if isinstance(v, float) and v > 100 else '', subset=['Current on Hand'])

st.dataframe(styled_df, height=300)



# Simulation plot
st.markdown("### Inventory Simulation Forecast (Next 12 Months)")
fig, ax = plt.subplots(figsize=(15, 7))
for sku in df_filtered['SKU']:
    sku_sim = sim_inv_df[sim_inv_df['SKU'] == sku]
    ax.plot(sku_sim['Month'], sku_sim['On Hand'], label=sku)
threshold_stock = df_filtered['Average Transfer'].min() * ((lead_time_days + reorder_buffer_days) / 30)
ax.axhline(y=threshold_stock, color='red', linestyle='--', label=f'Reorder Threshold ({lead_time_days + reorder_buffer_days} days stock)')
ax.set_xticks(range(len(forecast_months)))
ax.set_xticklabels(forecast_months, rotation=45)
ax.set_ylabel("Inventory Level (units)")
ax.set_title("Projected Inventory Levels by SKU")
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.markdown("### Yearly Reorder Quantities by SKU")

historical_df = pd.DataFrame(cost)
historical_df['Date'] = pd.to_datetime(historical_df['Date'])
historical_df['Year'] = historical_df['Date'].dt.year
historical_df['Total'] = historical_df['Qty'] * historical_df['Unit Price']
# Group by Year and Size
hist_summary = historical_df.groupby(['Year', 'Size'], as_index=False).agg({
    'Qty': 'sum',
    'Total': 'sum'
})

# Plot
fig1, ax1 = plt.subplots(figsize=(10, 5))
bars = sns.barplot(data=hist_summary, x='Year', y='Qty', hue='Size', ax=ax1)

# Add data labels
for container in ax1.containers:
    ax1.bar_label(container, fmt='%.0f', label_type='edge', padding=3)

# Final touches
ax1.set_title("Historical Purchase Quantity per Year")
ax1.set_ylabel("Quantity")
st.pyplot(fig1)

# Aggregate reorder quantities by SKU
reorder_qty_summary = sim_reorder_df.groupby('SKU')['Reorder Qty'].sum().reset_index()

# Plot bar chart
fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
sns.barplot(data=reorder_qty_summary, x='SKU', y='Reorder Qty', palette='viridis', ax=ax_bar)
ax_bar.set_title("Total Reorder Quantities Over Forecast Year")
ax_bar.set_xlabel("SKU")
ax_bar.set_ylabel("Reorder Quantity (units)")
for i, v in enumerate(reorder_qty_summary['Reorder Qty']):
    ax_bar.text(i, v + max(reorder_qty_summary['Reorder Qty']) * 0.01, f"{int(v)}", ha='center', va='bottom')
ax_bar.grid(axis='y')
st.pyplot(fig_bar)



# --- Estimated Purchasing Budget (Next 12 Months) ---

# Convert your 'data' dictionary to a DataFrame
cp_df = pd.DataFrame(data)[['SKU', 'Current Price (CP)']]

# Merge reorder quantities with CP
budget_df = pd.merge(reorder_qty_summary, cp_df, on='SKU', how='left')

# Calculate estimated spend per SKU
budget_df['Estimated Spend'] = budget_df['Reorder Qty'] * budget_df['Current Price (CP)']

# Total estimated spend
total_budget = budget_df['Estimated Spend'].sum()

# Optional: Bar plot of spend by SKU
fig_budget, ax_budget = plt.subplots(figsize=(10, 5))
sns.barplot(data=budget_df, x='SKU', y='Estimated Spend', palette='magma', ax=ax_budget)
ax_budget.set_title("Estimated Spend by SKU")
ax_budget.set_ylabel("Estimated Spend (R)")
for i, v in enumerate(budget_df['Estimated Spend']):
    ax_budget.text(i, v + 1, f"R{v:,.0f}", ha='center', va='bottom')
ax_budget.grid(axis='y')
st.pyplot(fig_budget)

# Show total in metric
st.metric("💸 Total Forecasted Spend", f"R{total_budget:,.2f}")



import datetime

# Assume forecast_months is sorted and your simulation covers these months
next_month = forecast_months[0]  # Or dynamically choose next month based on today

# Filter reorders scheduled for the next month
reorders_next_month = sim_reorder_df[sim_reorder_df['Month'] == next_month]

# Prepare reorder advice DataFrame
advice_rows = []

for sku in selected_skus:
    # Get current stock from simulation for next month
    inv_row = sim_inv_df[(sim_inv_df['SKU'] == sku) & (sim_inv_df['Month'] == next_month)]
    current_stock = inv_row['On Hand'].values[0] if not inv_row.empty else np.nan

    # Check if reorder scheduled next month
    reorder_row = reorders_next_month[reorders_next_month['SKU'] == sku]

    if not reorder_row.empty:
        reorder_qty = reorder_row['Reorder Qty'].values[0]
        status = "Order NOW"
    else:
        reorder_qty = 0
        status = "No reorder needed"

    advice_rows.append({
        'SKU': sku,
        'Projected On Hand': round(current_stock, 1),
        'Reorder Qty': reorder_qty,
        'Reorder Status': status
    })

reorder_advice_df = pd.DataFrame(advice_rows)

# Highlight function for urgent orders
def highlight_reorder(val):
    if isinstance(val, str) and 'NOW' in val:
        return 'color: red; font-weight: bold'
    return ''

st.markdown("### Reorder Advice for Next Month")
st.dataframe(reorder_advice_df.style.applymap(highlight_reorder, subset=['Reorder Status']), height=250)

# Export reorder advice CSV button
csv = reorder_advice_df.to_csv(index=False).encode('utf-8')
st.download_button("Download Reorder Advice CSV", csv, "reorder_advice.csv", "text/csv")


import statsmodels.api as sm

st.markdown("---")
st.markdown("### Sales Overview")

# Seasonal decomposition
st.markdown("### Seasonal Decomposition of Sales")

# Convert Month strings to datetime for decomposition
sales_df['Month_dt'] = pd.to_datetime(sales_df['Month'], format='%b-%y')
sales_ts = sales_df.set_index('Month_dt')['Sales']

# Perform seasonal decomposition (multiplicative model assumed)
decomposition = sm.tsa.seasonal_decompose(sales_ts, model='multiplicative', period=12)

fig_decomp, axs = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
decomposition.observed.plot(ax=axs[0], color='blue', legend=False)
axs[0].set_ylabel('Observed')
axs[0].set_title('Observed')

decomposition.trend.plot(ax=axs[1], color='orange', legend=False)
axs[1].set_ylabel('Trend')
axs[1].set_title('Trend')

decomposition.seasonal.plot(ax=axs[2], color='green', legend=False)
axs[2].set_ylabel('Seasonal')
axs[2].set_title('Seasonal')

decomposition.resid.plot(ax=axs[3], color='red', legend=False)
axs[3].set_ylabel('Residual')
axs[3].set_title('Residual')

plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig_decomp)

st.markdown("### Alerts & Recommendations with Actual Values and % Distance from Goals")

alert_rows = []

for idx, row in df_filtered.iterrows():
    sku = row['SKU']

    # Calculate % difference: ((actual - goal) / goal) * 100
    # DOH lower is better, so invert sign
    to_diff = ((row['T/O'] - goal_to) / goal_to) * 100
    doh_diff = ((goal_doh - row['DOH']) / goal_doh) * 100
    gmroi_diff = ((row['GMROI'] - goal_gmroi) / goal_gmroi) * 100
    str_diff = ((row['Sell Through Rate'] - goal_str) / goal_str) * 100

    alerts_for_sku = {
        'SKU': sku,

        # Actual values
        'Turnover (T/O)': f"{row['T/O']:.2f}",
        'Days on Hand (DOH)': f"{row['DOH']:.1f}",
        'GMROI': f"{row['GMROI']:.2f}",
        'Sell Through Rate (%)': f"{row['Sell Through Rate']:.1f}",
        'Current Stock': int(row['Current on Hand']),
        'Reorder Point (units)': int(row['Reorder Point (units)']),

        # Alerts (✅ or ⚠️)
        'Stock ≥ Reorder Point': '✅' if row['Current on Hand'] >= row['Reorder Point (units)'] else '⚠️',
        'Turnover ≥ Goal': '✅' if row['T/O'] >= goal_to else '⚠️',
        'DOH ≤ Goal': '✅' if row['DOH'] <= goal_doh else '⚠️',
        'GMROI ≥ Goal': '✅' if row['GMROI'] >= goal_gmroi else '⚠️',
        'Sell Thru Rate ≥ Goal': '✅' if row['Sell Through Rate'] >= goal_str else '⚠️',

        # Percentage differences with sign
        '% from Turnover Goal': f"{to_diff:+.1f}%",
        '% from DOH Goal': f"{doh_diff:+.1f}%",
        '% from GMROI Goal': f"{gmroi_diff:+.1f}%",
        '% from Sell Thru Rate Goal': f"{str_diff:+.1f}%",
    }
    alert_rows.append(alerts_for_sku)

alerts_df = pd.DataFrame(alert_rows)

def color_alerts(val):
    if val == '⚠️':
        return 'color: red; font-weight: bold'
    elif val == '✅':
        return 'color: green; font-weight: bold'
    return ''

def color_pct(val):
    try:
        num = float(val.replace('%', ''))
        if num < 0:
            return 'color: red; font-weight: bold'
        else:
            return 'color: green; font-weight: bold'
    except:
        return ''

# Columns with alerts and percentages to style
alert_cols = [
    'Stock ≥ Reorder Point', 'Turnover ≥ Goal', 'DOH ≤ Goal',
    'GMROI ≥ Goal', 'Sell Thru Rate ≥ Goal'
]
pct_cols = [
    '% from Turnover Goal', '% from DOH Goal', '% from GMROI Goal', '% from Sell Thru Rate Goal'
]

styled_df = alerts_df.style.applymap(color_alerts, subset=alert_cols)\
                            .applymap(color_pct, subset=pct_cols)

st.dataframe(styled_df)

# Step 2: Initialize an empty list to store the reorder dates for each SKU
reorder_dates = []

# Step 3: Loop through each SKU and check when inventory falls below the threshold
for sku in df_filtered['SKU']:
    # Filter the data for the current SKU
    sku_sim = sim_inv_df[sim_inv_df['SKU'] == sku]
    
    # Find rows where inventory goes below the threshold
    reorder_dates_for_sku = sku_sim[sku_sim['On Hand'] < threshold_stock]
    
    # Store reorder dates (assumed to be 'Month' here)
    reorder_dates_for_sku = reorder_dates_for_sku[['Month']].rename(columns={'Month': 'Reorder Date'})
    
    # Add the SKU column to the result
    reorder_dates_for_sku['SKU'] = sku
    
    # Append to the main list
    reorder_dates.append(reorder_dates_for_sku)

# Step 4: Concatenate all results into a single dataframe
reorder_dates_df = pd.concat(reorder_dates).reset_index(drop=True)

# Step 5: Group by SKU and aggregate reorder dates as a list
reorder_dates_per_sku = reorder_dates_df.groupby('SKU')['Reorder Date'].apply(list).reset_index()

# Step 6: Expand the list of reorder dates into separate columns
max_reorders = reorder_dates_per_sku['Reorder Date'].apply(len).max()

# Create new columns for reorder dates, with NaN where there are no reorder dates
for i in range(max_reorders):
    reorder_dates_per_sku[f'Reorder Date {i+1}'] = reorder_dates_per_sku['Reorder Date'].apply(lambda x: x[i] if i < len(x) else None)

# Drop the 'Reorder Date' column (since we now have individual date columns)
reorder_dates_per_sku = reorder_dates_per_sku.drop(columns='Reorder Date')

# Step 7: Display the table in Streamlit
st.markdown("### Reorder Dates for Each SKU")
# Display the dataframe as a neat table in Streamlit
st.dataframe(reorder_dates_per_sku)

# ========================= 📊 SALES PERFORMANCE & INVENTORY DASHBOARD =========================

from datetime import date
from prophet import Prophet

st.markdown("## 📊 Sales & Inventory Performance Dashboard")

# --- PREPARE DATA ---
sales_df['Month_dt'] = pd.to_datetime(sales_df['Month'], format='%b-%y')
sales_df['Year'] = sales_df['Month_dt'].dt.year
sales_df = sales_df.sort_values('Month_dt')
sales_df['Month_Name'] = sales_df['Month_dt'].dt.strftime('%b')
sales_df['Quarter'] = sales_df['Month_dt'].dt.to_period('Q')

# --- METRIC VARIABLES ---
current_year = date.today().year
previous_year = current_year - 1
current_month = date.today().month

# Restrict to only months available this year
year_months = sales_df[sales_df['Year'] == current_year]['Month_dt'].dt.month.unique()
sales_ytd_current = sales_df[(sales_df['Year'] == current_year) & (sales_df['Month_dt'].dt.month.isin(year_months))]
sales_ytd_previous = sales_df[(sales_df['Year'] == previous_year) & (sales_df['Month_dt'].dt.month.isin(year_months))]

# --- KPI METRICS ---
ytd_total_current = sales_ytd_current['Sales'].sum()
ytd_total_previous = sales_ytd_previous['Sales'].sum()
ytd_avg_monthly = ytd_total_current / len(year_months) if len(year_months) > 0 else 0

yoy_diff = ytd_total_current - ytd_total_previous
yoy_pct = (yoy_diff / ytd_total_previous * 100) if ytd_total_previous > 0 else 0

sales_df['MoM % Change'] = sales_df['Sales'].pct_change() * 100
latest_month = sales_df['Month'].iloc[-1]
latest_sales = sales_df['Sales'].iloc[-1]
latest_mom = sales_df['MoM % Change'].iloc[-1]

sales_df['Cumulative Sales'] = sales_df['Sales'].cumsum()
sales_df['Rolling Avg (3m)'] = sales_df['Sales'].rolling(window=3).mean()

# --- SPEND METRICS ---
cost_df = pd.DataFrame(cost_tot)
cost_df['Date'] = pd.to_datetime(cost_df['Date'])
cost_df['Year'] = cost_df['Date'].dt.year
cost_df['Month'] = cost_df['Date'].dt.month
cost_df['Total Spend'] = cost_df['Qty'] * cost_df['Unit Price']

spend_ytd_current = cost_df[(cost_df['Year'] == current_year) & (cost_df['Month'].isin(year_months))]
spend_ytd_previous = cost_df[(cost_df['Year'] == previous_year) & (cost_df['Month'].isin(year_months))]

spend_total_current = spend_ytd_current['Total Spend'].sum()
spend_total_previous = spend_ytd_previous['Total Spend'].sum()
spend_diff = spend_total_current - spend_total_previous
spend_pct = (spend_diff / spend_total_previous * 100) if spend_total_previous > 0 else 0

# ========================= 📌 KPI METRICS =========================
col1, col2, col3, col4 = st.columns(4)

# --- KPI 1: YTD Sales ---
ytd_delta_color = "normal" if ytd_total_current >= ytd_total_previous else "inverse"
ytd_delta_text = f"▲ R{abs(ytd_total_current - ytd_total_previous):,.0f}" if ytd_total_current >= ytd_total_previous else f"▼ R{abs(ytd_total_current - ytd_total_previous):,.0f}"
col1.metric(
    label="📅 YTD Sales",
    value=f"R{ytd_total_current:,.0f}",
    delta=ytd_delta_text,
    delta_color=ytd_delta_color
)

# --- KPI 2: YoY Growth ---
yoy_delta_color = "normal" if yoy_pct >= 0 else "inverse"
yoy_delta_text = f"▲ R{abs(yoy_diff):,.0f}" if yoy_pct >= 0 else f"▼ R{abs(yoy_diff):,.0f}"
col2.metric(
    label="📊 YoY Growth (Same Months)",
    value=f"{yoy_pct:+.1f}%",
    delta=yoy_delta_text,
    delta_color=yoy_delta_color
)

# --- KPI 3: Latest Month Sales ---
mom_delta_color = "normal" if latest_mom >= 0 else "inverse"
mom_delta_text = f"▲ {abs(latest_mom):.1f}%" if latest_mom >= 0 else f"▼ {abs(latest_mom):.1f}%"
col3.metric(
    label=f"📈 Latest Month: {latest_month}",
    value=f"R{latest_sales:,.0f}",
    delta=mom_delta_text,
    delta_color=mom_delta_color
)

# --- KPI 4: YTD Spend ---
spend_delta_color = "inverse" if spend_total_current > spend_total_previous else "normal"
spend_delta_text = f"▲  R{abs(spend_diff):,.0f}" if spend_total_current > spend_total_previous else f"▼ R{abs(spend_diff):,.0f}"

col4.metric(
    label="💸 YTD Spend",
    value=f"R{spend_total_current:,.0f}",
    delta=spend_delta_text,
    delta_color=spend_delta_color
)

# ========================= 📊 SALES TREND CHART =========================

# --- Preprocessing ---
cost_tot['Date'] = pd.to_datetime(cost_tot['Date'], dayfirst=False)
cost_tot['Year'] = cost_tot['Date'].dt.year
cost_yearly = cost_tot.groupby('Year')['Total'].sum().reset_index()
cost_yearly = cost_yearly.rename(columns={'Total': 'Cost'})

sales_df['Month_dt'] = pd.to_datetime(sales_df['Month'], format='%b-%y')
sales_df['Year'] = sales_df['Month_dt'].dt.year
sales_yearly = sales_df.groupby('Year')['Sales'].sum().reset_index()

# --- Merge and Compute GP ---
summary = pd.merge(sales_yearly, cost_yearly, on='Year', how='left')
summary['Cost'] = summary['Cost'].fillna(0)
summary['GP'] = summary['Sales'] - summary['Cost']
summary['GP %'] = (summary['GP'] / summary['Sales']) * 100

# --- Format Values for Display ---
summary_display = summary.copy()
summary_display['Sales'] = summary_display['Sales'].apply(lambda x: f"R{x:,.0f}")
summary_display['Cost'] = summary_display['Cost'].apply(lambda x: f"R{x:,.0f}")
summary_display['GP'] = summary_display['GP'].apply(lambda x: f"R{x:,.0f}")
summary_display['GP %'] = summary_display['GP %'].apply(lambda x: f"{x:.1f}%")

# --- Display Table ---
st.markdown("## 🧾 Yearly Gross Profit Summary")
st.dataframe(summary_display.rename(columns={
    'Year': '📅 Year',
    'Sales': '💰 Sales',
    'Cost': '📦 Cost',
    'GP': '📈 GP',
    'GP %': '📊 GP %'
}), use_container_width=True)


# ========================= 💰 PROFIT VS SALES =========================
import matplotlib.patches as mpatches

st.markdown("## 💰 Profit vs Sales Analysis")

# --- Prepare monthly sales and cost data ---
cost_df = pd.DataFrame(cost_tot)
cost_df['Date'] = pd.to_datetime(cost_df['Date'])
cost_df['YearMonth'] = cost_df['Date'].dt.to_period('M')
cost_df['Total Cost'] = cost_df['Qty'] * cost_df['Unit Price']

# Aggregate monthly sales and cost
sales_monthly = sales_df.groupby(sales_df['Month_dt'].dt.to_period('M'))['Sales'].sum().reset_index()
sales_monthly['Month_dt'] = sales_monthly['Month_dt'].dt.to_timestamp()
cost_monthly = cost_df.groupby('YearMonth')['Total Cost'].sum().reset_index()
cost_monthly['YearMonth'] = cost_monthly['YearMonth'].dt.to_timestamp()

# Merge sales and cost
profit_df = pd.merge(sales_monthly, cost_monthly, left_on='Month_dt', right_on='YearMonth', how='left')
profit_df['Total Cost'] = profit_df['Total Cost'].fillna(0)
profit_df['Profit'] = profit_df['Sales'] - profit_df['Total Cost']
profit_df['GP %'] = profit_df['Profit'] / profit_df['Sales'] * 100

# --- Plot Profit vs Sales with GP% and highlights ---
fig, ax = plt.subplots(figsize=(12, 6))

# Colors
sales_color = '#0072B2'       # Blue
profit_color = '#009E73'      # Green
gp_color = '#D55E00'          # Orange
low_gp_color = '#E69F00'      # Orange/yellow for GP% < 30%
cost_exceed_color = '#CC79A7' # Pink/purple for Cost > Sales
record_color = '#F0E442'      # Yellow for record profit

# Plot Sales and Profit lines
ax.plot(profit_df['Month_dt'], profit_df['Sales'], marker='o', label='Sales', color=sales_color, linewidth=2)
ax.plot(profit_df['Month_dt'], profit_df['Profit'], marker='o', label='Profit', color=profit_color, linewidth=2)

# Priority highlights
threshold = 30  # GP% threshold
for idx, row in profit_df.iterrows():
    highlight_color = None
    if row['Total Cost'] > row['Sales']:
        highlight_color = cost_exceed_color
    elif row['GP %'] < threshold:
        highlight_color = low_gp_color
    if highlight_color:
        ax.axvspan(row['Month_dt'] - pd.Timedelta(days=15),
                   row['Month_dt'] + pd.Timedelta(days=15),
                   color=highlight_color, alpha=0.2)

# Add trophy/star for best month of each year
for year, group in profit_df.groupby(profit_df['Month_dt'].dt.year):
    best_idx = group['Profit'].idxmax()
    best_row = profit_df.loc[best_idx]
    best_date = best_row['Month_dt']
    best_profit = best_row['Profit']
    # Gold star marker
    ax.scatter(best_date, best_profit * 1.05, marker="*", s=200, color="gold", zorder=5, label="Best Month" if year == profit_df['Month_dt'].dt.year.min() else "")

# Secondary axis for GP %
ax2 = ax.twinx()
ax2.plot(profit_df['Month_dt'], profit_df['GP %'], marker='x', linestyle='--', color=gp_color, label='GP %')
ax2.set_ylabel("Gross Profit %", color=gp_color)
ax2.tick_params(axis='y', labelcolor=gp_color)
ax2.set_ylim(0, 100)

# Labels, grid, and combined legend
ax.set_title("Monthly Profit vs Sales with Key Highlights")
ax.set_xlabel("Month")
ax.set_ylabel("Amount (ZAR)")
ax.grid(True)

# Legend patches
low_gp_patch = mpatches.Patch(color=low_gp_color, alpha=0.2, label='GP% < 30%')
cost_exceed_patch = mpatches.Patch(color=cost_exceed_color, alpha=0.2, label='Cost > Sales')

lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(
    lines + lines2 + [low_gp_patch, cost_exceed_patch],
    labels + labels2 + [low_gp_patch.get_label(), cost_exceed_patch.get_label()],
    loc='upper left',
    bbox_to_anchor=(1.02, 1),
    borderaxespad=0
)

plt.xticks(rotation=45)
st.pyplot(fig)

# --- Table View ---
st.markdown("### Profit & Sales Table")
profit_df_display = profit_df.copy()
profit_df_display['Sales'] = profit_df_display['Sales'].map(lambda x: f"R{x:,.0f}")
profit_df_display['Total Cost'] = profit_df_display['Total Cost'].map(lambda x: f"R{x:,.0f}")
profit_df_display['Profit'] = profit_df_display['Profit'].map(lambda x: f"R{x:,.0f}")
profit_df_display['GP %'] = profit_df_display['GP %'].map(lambda x: f"{x:.1f}%")

# Flags for table
profit_df_display['Cost > Sales'] = profit_df['Total Cost'] > profit_df['Sales']
profit_df_display['Cost > Sales'] = profit_df_display['Cost > Sales'].map(lambda x: "⚠️" if x else "")
profit_df_display['Low GP%'] = profit_df['GP %'] < threshold
profit_df_display['Low GP%'] = profit_df_display['Low GP%'].map(lambda x: "⬇️" if x else "")

# Add record profit (trophy) in table
for year, group in profit_df.groupby(profit_df['Month_dt'].dt.year):
    best_idx = group['Profit'].idxmax()
    profit_df_display.loc[best_idx, 'Record Profit'] = "🏆"
profit_df_display['Record Profit'] = profit_df_display.get('Record Profit', "")

st.dataframe(profit_df_display[['Month_dt', 'Sales', 'Total Cost', 'Profit', 'GP %', 'Cost > Sales', 'Low GP%', 'Record Profit']], use_container_width=True)



# ========================= 📉 6-MONTH COMPARATIVE SUMMARY =========================
st.markdown("---")
st.markdown("### 📉 12-Month Comparative Summary (with YoY Growth)")

# --- Get the last 12 distinct months from the data ---
last_12_months = sales_df['Month_dt'].drop_duplicates().nlargest(12).sort_values()
recent_month_nums = last_12_months.dt.month.unique()
month_names = last_12_months.dt.strftime('%b').unique()

# --- Filter dataset for only those 6 months across all years ---
filtered_df = sales_df[sales_df['Month_dt'].dt.month.isin(recent_month_nums)].copy()
filtered_df['Month_Name'] = filtered_df['Month_dt'].dt.strftime('%b')

# --- Group by Month Name and Year ---
grouped_df = filtered_df.groupby(['Month_Name', 'Year'])['Sales'].sum().reset_index()

# --- Pivot to create summary table: rows = Month, columns = Years ---
summary_6m = grouped_df.pivot(index='Month_Name', columns='Year', values='Sales').reset_index()

# --- Sort months in correct calendar order ---
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
summary_6m['Month_Num'] = summary_6m['Month_Name'].apply(lambda m: month_order.index(m))
summary_6m = summary_6m.sort_values('Month_Num').drop(columns='Month_Num')

# --- Add YoY % growth columns ---
years = sorted([col for col in summary_6m.columns if isinstance(col, int)])
for i in range(1, len(years)):
    curr_year = years[i]
    prev_year = years[i - 1]
    growth_col = f"{prev_year}→{curr_year} %"
    summary_6m[growth_col] = (summary_6m[curr_year] - summary_6m[prev_year]) / summary_6m[prev_year] * 100

# --- Format values ---
for col in summary_6m.columns:
    if isinstance(col, int):
        summary_6m[col] = summary_6m[col].map(lambda x: f"R{x:,.0f}" if pd.notnull(x) else "—")
    elif '→' in col:
        summary_6m[col] = summary_6m[col].map(lambda x: f"{x:+.1f}%" if pd.notnull(x) else "—")

# --- Display table ---
st.dataframe(summary_6m, use_container_width=True)



# ========================= 📊 YTD COMPARISON TABLE =========================
st.markdown("### 📊 Year-to-Date Sales Comparison (Same Period per Year)")

# --- Prep: Convert to datetime and extract parts
sales_df['Month_dt'] = pd.to_datetime(sales_df['Month_dt'])
sales_df['Year'] = sales_df['Month_dt'].dt.year
sales_df['Month'] = sales_df['Month_dt'].dt.month

# --- Step 1: Identify the last fully completed month in current year
today = pd.Timestamp.today()
current_year = today.year
last_complete_month = today.month - 1 if today.day < 28 else today.month - 1
if last_complete_month == 0:
    last_complete_month = 12
    current_year -= 1

# --- Step 2: Filter data to only months from Jan to last_complete_month
valid_months = list(range(1, last_complete_month + 1))
filtered_df = sales_df[sales_df['Month'].isin(valid_months)]

# --- Step 3: Ensure we include only data up to that month for all years
ytd_compare_df = (
    filtered_df.groupby('Year')['Sales']
    .agg(['sum', 'count'])
    .rename(columns={'sum': 'Total Sales', 'count': 'Months Count'})
    .reset_index()
)

# --- Step 4: Calculate average
ytd_compare_df['Avg Monthly'] = ytd_compare_df['Total Sales'] / ytd_compare_df['Months Count']

# --- Step 5: Format nicely
ytd_compare_df['Total Sales'] = ytd_compare_df['Total Sales'].map(lambda x: f"R{x:,.0f}")
ytd_compare_df['Avg Monthly'] = ytd_compare_df['Avg Monthly'].map(lambda x: f"R{x:,.0f}")
ytd_compare_df = ytd_compare_df.drop(columns='Months Count')

# --- Step 6: Display
st.dataframe(ytd_compare_df, use_container_width=True)


# Plot the data
# Convert formatted strings back to numeric for plotting
plot_df = ytd_compare_df.copy()
plot_df['Total Sales'] = plot_df['Total Sales'].replace('[R,]', '', regex=True).astype(float)

# Create figure
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(plot_df['Year'].astype(str), plot_df['Total Sales'])

# Title with dynamic month
ax.set_title(f"YTD Sales Comparison (Jan–{pd.to_datetime(last_complete_month, format='%m').strftime('%b')})", fontsize=14)
ax.set_ylabel('Total Sales (R)', fontsize=12)
ax.set_xlabel('Year', fontsize=12)

# Format ticks
ax.tick_params(axis='x', labelrotation=0)
ax.tick_params(axis='y', labelsize=10)

# Add value labels
ax.bar_label(bars, fmt='R%.0f', fontsize=9, padding=3)

# Show in Streamlit
st.pyplot(fig)

# ========================= 📜 MONTHLY SALES TABLE =========================
import matplotlib.pyplot as plt

# --- 📜 Monthly Sales & Growth Table ---
st.markdown("### 📜 Monthly Sales & Growth Table")

# Format table for display
table_df = sales_df[['Month_dt', 'Sales']].copy()
table_df['Month'] = table_df['Month_dt'].dt.strftime('%b %Y')
table_df['MoM % Change'] = sales_df['Sales'].pct_change() * 100
table_df['Sales'] = table_df['Sales'].map(lambda x: f"R{x:,.0f}")
table_df['MoM % Change'] = table_df['MoM % Change'].map(lambda x: f"{x:+.1f}%" if pd.notnull(x) else "—")

# Reorder and display
table_df = table_df[['Month', 'Sales', 'MoM % Change']]
st.dataframe(table_df, use_container_width=True)

# --- 📈 MoM % Change Graph ---
st.markdown("### 📈 MoM % Change in Sales (Oct 2020 – Jul 2025)")

# Prepare data for plotting
plot_df = sales_df[['Month_dt', 'Sales']].copy()
plot_df = plot_df.sort_values('Month_dt')
plot_df['MoM % Change'] = plot_df['Sales'].pct_change() * 100

# Limit range from Oct 2020 to Jul 2025
plot_df = plot_df[(plot_df['Month_dt'] >= '2020-10-01') & (plot_df['Month_dt'] <= '2025-08-30')]

# Plot
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(plot_df['Month_dt'], plot_df['MoM % Change'], marker='o', linestyle='-')

# Formatting
ax.set_title("Month-over-Month % Change in Sales (Oct 2020 – Jul 2025)")
ax.set_ylabel("MoM % Change (%)")
ax.set_xlabel("Month")
ax.axhline(0, color='gray', linewidth=1, linestyle='--')
ax.grid(True)
plt.xticks(rotation=45)

# Show graph
st.pyplot(fig)


# ========================= 📊 SALES TREND BY YEAR =========================
st.markdown("### 📊 Sales Trends by Year")
fig_yearly, ax = plt.subplots(figsize=(12, 5))
for year, group in sales_df.groupby('Year'):
    ax.plot(group['Month_dt'], group['Sales'], label=str(year), marker='o')
ax.set_title("Monthly Sales Trend (by Year)")
ax.set_ylabel("Sales (ZAR)")
ax.set_xlabel("Month")
ax.legend()
ax.grid(True)
plt.xticks(rotation=45)
st.pyplot(fig_yearly)

# ========================= 📉 ROLLING AVERAGE CHART =========================
st.markdown("### 📉 3-Month Rolling Average vs Actual Sales")
fig_roll, ax_roll = plt.subplots(figsize=(12, 5))
ax_roll.plot(sales_df['Month_dt'], sales_df['Sales'], label='Monthly Sales', alpha=0.4)
ax_roll.plot(sales_df['Month_dt'], sales_df['Rolling Avg (3m)'], label='Rolling Avg (3m)', linewidth=2)
ax_roll.set_title("Sales with 3-Month Rolling Average")
ax_roll.set_ylabel("Sales (ZAR)")
ax_roll.set_xlabel("Month")
ax_roll.legend()
ax_roll.grid(True)
plt.xticks(rotation=45)
st.pyplot(fig_roll)

# ========================= 🪮 QUARTERLY PERFORMANCE =========================
st.markdown("### 🪮 Quarterly Sales Growth by Quarter (All Years)")

# --- Make sure these columns exist and are correct ---
sales_df['Year'] = sales_df['Month_dt'].dt.year
sales_df['Quarter'] = sales_df['Month_dt'].dt.quarter

# --- Aggregate quarterly data ---
quarter_df = sales_df.groupby(['Year', 'Quarter'])['Sales'].sum().reset_index()

# ✅ Convert Quarter to 'Q1', 'Q2', etc. BEFORE setting it as a category
quarter_df['Quarter'] = 'Q' + quarter_df['Quarter'].astype(str)

# ✅ Set correct order for quarters
quarter_order = pd.api.types.CategoricalDtype(categories=['Q1', 'Q2', 'Q3', 'Q4'], ordered=True)
quarter_df['Quarter'] = quarter_df['Quarter'].astype(quarter_order)

# ✅ Convert Year to string for hue
quarter_df['Year'] = quarter_df['Year'].astype(str)

# --- Debugging tip: print to check ---
# st.dataframe(quarter_df)  # Uncomment this line to check if quarter_df is populated correctly

# --- Plot ---
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=quarter_df, x='Quarter', y='Sales', hue='Year', ax=ax)

# --- Styling ---
ax.set_title("Quarterly Sales Comparison Across Years", fontsize=16)
ax.set_xlabel("Quarter", fontsize=13)
ax.set_ylabel("Total Sales (R)", fontsize=13)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.legend(title='Year', title_fontsize=12, fontsize=11, bbox_to_anchor=(1.01, 1), loc='upper left')

# Optional: add value labels
for container in ax.containers:
    ax.bar_label(
        container,
        fmt='R%.0f',
        fontsize=8,
        label_type='edge',
        padding=2,
        rotation=90  # 👈 Rotate labels 45 degrees (you can try 90 or other angles too)
    )


# --- Show chart ---
st.pyplot(fig)




# ========================= 🔮 SALES FORECASTING =========================
# --- Prepare Data ---
from pmdarima import auto_arima
import pandas as pd
import matplotlib.pyplot as plt

# --- Prepare data ---
sales_df = sales_df.copy()
sales_df = sales_df.set_index('Month_dt').asfreq('MS')  # Ensure monthly frequency

model = auto_arima(
    sales_df['Sales'],
    seasonal=True,
    m=12,
    stepwise=True,
    suppress_warnings=True,
    trace=True,
    information_criterion='aic'  # use BIC instead of AIC
)


# --- Forecast ---
n_periods = 3
forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)

forecast_index = pd.date_range(start=sales_df.index[-1] + pd.offsets.MonthBegin(1), periods=n_periods, freq='MS')
forecast_df = pd.DataFrame({
    'Month': forecast_index,
    'Predicted Sales': forecast,
    'Lower Bound': conf_int[:, 0],
    'Upper Bound': conf_int[:, 1]
}).round(0).astype({'Predicted Sales': int, 'Lower Bound': int, 'Upper Bound': int})
forecast_df['Month'] = forecast_df['Month'].dt.strftime('%B %Y')

# --- Plot ---
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(sales_df.index, sales_df['Sales'], label='Historical Sales')
ax.plot(forecast_index, forecast, label='Forecast')
ax.fill_between(forecast_index, conf_int[:, 0], conf_int[:, 1], color='gray', alpha=0.3)
ax.set_title("Sales Forecast (AutoARIMA)")
ax.legend()
st.pyplot(fig)

# --- Summary ---
st.metric(
    label="🔮 3-Month Forecast Total (AutoARIMA)",
    value=f"R{forecast_df['Predicted Sales'].sum():,.0f}",
    delta=f"Range: R{forecast_df['Lower Bound'].sum():,.0f} – R{forecast_df['Upper Bound'].sum():,.0f}"
)

# --- Table ---
st.markdown("#### 📈 Forecasted Monthly Sales")
st.dataframe(forecast_df, use_container_width=True)



# === Footer Section ===
# Footer
st.markdown("---")
st.markdown("Developed by Godwin & co - Interactive Inventory Dashboard")
