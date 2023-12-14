import pandas as pd 

data = pd.read_csv("registro_metricas_ventanas.csv")
data['month&day'] = data['date'].apply(lambda x: x[5:])
data['year'] = data['date'].apply(lambda x: x[:4])

import matplotlib.pyplot as plt
import seaborn as sns

def plot_time_series(valor, data):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=data['month&day'], y=data[valor], color='blue', hue=data['year'])
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.DayLocator(interval=15))
    plt.gcf().set_facecolor('white')
    plt.xlabel('Fecha')
    plt.ylabel('Valor')
    plt.title(f'{valor}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/{valor}.png')
    
for col in data.columns:
    plot_time_series(col, data)