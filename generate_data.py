"""Creates the sample dataset (CSV files + vendor documents). Run once:  python generate_data.py"""
import numpy as np, pandas as pd, json, os, math, datetime as dt
rng = np.random.default_rng(42)
AS_OF = dt.date(2026, 11, 1)          # demo "today": the 11.11 sale is 10 days away
D = "data"; os.makedirs(f"{D}/vendor_docs", exist_ok=True)

# id, name, category, current_stock, unit_cost, selling_price, base_daily_sales
P = [("P01","Power Bank 10000mAh","Power",40,1900,2599,5.5),
     ("P02","Wireless Earbuds","Audio",65,2800,3799,4.0),
     ("P03","Phone Case (assorted)","Accessories",600,180,399,14),
     ("P04","USB-C Cable 1m","Accessories",110,220,449,11),
     ("P05","Screen Protector","Accessories",900,60,199,18),
     ("P06","Bluetooth Speaker","Audio",22,3400,4699,2.5),
     ("P07","Smart Watch","Wearables",150,4200,5499,2.2),
     ("P08","Car Charger","Power",420,450,799,3.5)]
products = pd.DataFrame(P, columns=["product_id","name","category","current_stock","unit_cost","selling_price","base_daily_sales"])
products[["product_id","name","category","current_stock","unit_cost","selling_price"]].to_csv(f"{D}/products.csv", index=False)

# events: name, start, days, multiplier (past ones only shape the history)
events = pd.DataFrame([("Independence Day Sale","2026-08-14",1,1.6),("9.9 Sale","2026-09-09",3,2.2),
                       ("11.11 Mega Sale","2026-11-11",3,2.5),("Black Friday","2026-11-27",3,2.0)],
                      columns=["name","start_date","days","multiplier"])
events.to_csv(f"{D}/events.csv", index=False)

# 150 days of daily sales history
days = pd.date_range(end=pd.Timestamp(AS_OF)-pd.Timedelta(days=1), periods=150)
rows = []
for _, p in products.iterrows():
    for d in days:
        m = 1.0
        for _, e in events.iterrows():
            s = pd.Timestamp(e.start_date)
            if s <= d < s + pd.Timedelta(days=int(e.days)): m = max(m, e.multiplier)
        wk = 1.15 if d.dayofweek in (4,5) else 1.0
        rows.append((d.date().isoformat(), p.product_id, int(rng.poisson(p.base_daily_sales*m*wk))))
pd.DataFrame(rows, columns=["date","product_id","units"]).to_csv(f"{D}/sales_history.csv", index=False)

# vendors
V = {"V1":("Lahore Gadget Hub","Lahore",1.04,20,7,0.94),
     "V2":("Karachi MegaTrade","Karachi",0.93,60,12,0.97),
     "V3":("Peshawar Tech Traders","Peshawar",1.00,10,4,0.95),
     "V4":("Rawalpindi Wholesale Co","Rawalpindi",0.97,80,9,0.92),
     "V5":("Faisalabad Imports","Faisalabad",1.00,30,8,0.96)}
SUP = {"P01":["V1","V2","V3"],"P02":["V1","V3","V4"],"P03":["V2","V4","V5"],"P04":["V3","V4","V5"],
       "P05":["V2","V4","V5"],"P06":["V1","V3","V5"],"P07":["V1","V4","V5"],"P08":["V2","V3","V4"]}
vr = []
for pid, vs in SUP.items():
    cost = float(products.loc[products.product_id==pid, "unit_cost"].iloc[0])
    tier = 5 if cost < 500 else (2.5 if cost < 2000 else 1)
    for v in vs:
        n, city, pf, mf, lead, flex = V[v]
        vr.append((v, n, city, pid, int(round(cost*pf)), int(math.ceil(mf*tier/10)*10),
                   max(2, lead + int(rng.integers(-1, 2))), "50% advance" if v in ("V2","V5") else "30 days credit", flex))
pd.DataFrame(vr, columns=["vendor_id","vendor_name","city","product_id","unit_price","moq","lead_days","payment_terms","flex_pct"]).to_csv(f"{D}/vendors.csv", index=False)

# vendor documents (plain text; RAG reads these). profile = (late invoices, damaged invoices, review tone)
PROFILE = {"V1":(0,0,"good"),"V2":(3,2,"bad"),"V3":(1,0,"ok"),"V4":(0,0,"good"),"V5":(1,2,"ok")}
REV = {"good":["Always delivers exactly when promised and packaging is excellent. Genuine products, very reliable supplier.",
               "Responsive on WhatsApp and honest about stock. Quality is consistent across batches.",
               "Five-star service. Replaced one faulty piece within two days without any argument."],
       "ok": ["Prices are fair and the owner is responsive, but one shipment arrived late during a busy season.",
              "Generally good quality. Packaging could be better; a few cartons arrived dented.",
              "Reliable most of the time. Communication is quick, delivery is average."],
       "bad":["Cheap but unreliable. Two orders were delivered late and we ran out of stock waiting.",
              "Several items were damaged and the supplier was slow to refund. Poor packaging.",
              "Delivery delay again this month. Difficult to reach on phone; I would not depend on them during sales."]}
for v,(n,city,*_) in V.items():
    late, dmg, tone = PROFILE[v]
    inv = []
    for i in range(6):
        l = f"Invoice INV-{v}-{101+i} | Order of {int(rng.integers(100,400))} units | "
        l += f"Delivered {int(rng.integers(3,9))} days late" if i < late else "Delivered on time"
        l += " | " + (f"Condition: {int(rng.integers(4,12))}% of items damaged on arrival" if i >= 6-dmg else "Condition: all items intact")
        inv.append(l)
    open(f"{D}/vendor_docs/{v}_invoices.txt","w").write(f"Past invoices and delivery records for {n} ({city})\n\n" + "\n\n".join(inv))
    open(f"{D}/vendor_docs/{v}_reviews.txt","w").write(f"Customer reviews of {n}\n\n" + "\n\n".join(REV[tone]))
    terms = {"V1":"Return policy: 7 days for defective items. Late delivery penalty: 2% discount per day.",
             "V2":"Return policy: 3 days only, restocking fee 10%. No late delivery penalty. Payment 50% in advance.",
             "V3":"Return policy: 5 days for defective items. Delivery within 4 days guaranteed for orders above minimum.",
             "V4":"Return policy: 7 days for defective items. Strict minimum order quantity, no exceptions. 30 days credit.",
             "V5":"Return policy: 5 days for defective items. Payment 50% in advance. Delivery times are estimates only."}[v]
    open(f"{D}/vendor_docs/{v}_contract.txt","w").write(f"Contract notes for {n}\n\n{terms}")
json.dump({"as_of": AS_OF.isoformat()}, open(f"{D}/meta.json","w"))
print("Data created in ./data")
