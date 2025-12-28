import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv(r"E:\Unemployment analytics\dataset\Unemployment_in_India.csv")
df.columns=df.columns.str.strip()
print("First 5 rows :\n")
print(df.head())
print("\nDataset Info:\n")
print(df.info())
print("\nMissing Values :\n")
print(df.isnull().sum())
#basic analysis
print("\nAverage Unemployment Rate:")
print(df['Estimated Unemployment Rate (%)'].mean())
state_avg =(df.groupby('Region')['Estimated Unemployment Rate (%)'].mean().sort_values(ascending=False))
print("\nState-wise Average Unemployment Rate:\n")
print(state_avg)
#visualizations
plt.figure()
state_avg.plot(kind='bar')
plt.title("Average Unemployment Rate by State")
plt.xlabel("State")
plt.ylabel("Unemployement Rate (%)")
plt.tight_layout()
plt.show()
