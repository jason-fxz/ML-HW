import pandas as pd
import matplotlib.pyplot as plt

# date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT

path = './datasets/ETTh1.csv'

start = 0
length_in = 168
length_out = 24


def show(data, start, length_in = 168, length_out = 24):


    data = data.iloc[start:start + length_in + length_out]
    plt.figure(figsize=(12, 6))
    for col in data.columns:
        if col != 'date':
            plt.plot(data['date'], data[col], label=col)
        print("col", col)
        
    plt.axvline(x=data['date'].iloc[length_in], color='red', linestyle='--', label='divide line')

    plt.xlabel('date')
    plt.ylabel('value')
    plt.title('ETTh1 Dataset')
    plt.legend()
    plt.xticks(data['date'][::12], rotation=45)
    plt.tight_layout()
    plt.show()

data = pd.read_csv(path)
print("readed\n", data.head())

show(data, 3444)



# save the figure
# plt.savefig('ETTh1_dataset.png')


# print(data.head())