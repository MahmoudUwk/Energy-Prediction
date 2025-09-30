import pandas as pd
from matplotlib import pyplot as plt

## Load the file (in this case the 1477227096132.csv)
data = pd.read_csv("C:/Users/mahmo/OneDrive/Desktop/kuljeet/pwr data paper 2/1Hz/1477227096132.csv")
data.set_index(pd.to_datetime(data.timestamp), inplace=True)
data.drop(columns=["timestamp"], inplace=True)

## Plot one day of voltage, current, active, and reactive power (re-scaled to the original values; resampled to 1Hz)
fig, axs = plt.subplots(4, 1, figsize = (20,10), sharex='col')

w = slice("2016-10-24", "2016-10-24")

axs[0].set_title("One Day of Voltage, Current, Active, and Reactive Power Re-sampled to 1 Hz")

axs[0].plot(data[w].V*460, color='r', label="Voltage") # 460 is the scaler for voltage
axs[0].plot([], color='b', label="Current") 
axs[0].plot([], color='g', label="Active Power")
axs[0].plot([], color='orange', label="Reactive Power")
axs[1].plot(data[w].I*52.5, color='b') # 52.5 is the scaler for current
axs[2].plot(data[w].P*460*52.5, color='g')
axs[3].plot(data[w].Q*460*52.5, color='orange')
axs[0].set_ylabel("Voltage (V)")
axs[1].set_ylabel("Current (A)")
axs[2].set_ylabel("Active P. (W)")
axs[3].set_ylabel("Reactive P. (VAR)")
axs[0].set_ylim([220,240])
axs[1].set_ylim([0,20])
axs[2].set_ylim([-100,2800])
axs[3].set_ylim([-1000,1000])

axs[3].set_xlabel("Samples")

axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.7), fancybox=True, shadow=True, ncol=4)
plt.tight_layout()
#plt.savefig("PQVI_1Hz.pdf")