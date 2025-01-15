import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_excel('C:/Users/asus/OneDrive/Documents/OneDrive/Desktop/Sheet2.xlsx')
freq = data['frequency']
formula = data['Formula']
print(formula)

plt.figure()
plt.semilogx(freq,formula,label = 'output')
plt.axhline(y=27.4,color = 'red',linestyle='--',linewidth =0.7,label = '{k}')
plt.grid(True,which='both',linestyle ='--',linewidth= 0.5)
differences = formula - 27.4
plt.xlabel('Frequency')
plt.ylabel('Gain(dB)')
plt.title('Frequency Bandwidth')
plt.legend()
# Find indices where the sign changes (indicating an intersection)
intersections = np.where(np.diff(np.sign(differences)))[0]

# Initialize lists to hold the intersection frequencies and voltages
intersection_freqs = []
intersection_voltages = []

# Calculate intersection points using linear interpolation
for idx in intersections:
    x1 = freq[idx]
    x2 = freq[idx + 1]
    y1 = formula[idx]
    y2 = formula[idx + 1]

    # Interpolate to find the exact intersection frequency
    intersection_freq = x1 + (x2 - x1) * (27.4 - y1) / (y2 - y1)
    intersection_freqs.append(intersection_freq)
    intersection_voltages.append(27.4)

# Print the intersection points
for f in intersection_freqs:
    print(f'Intersection at frequency: {f}')

for f in intersection_freqs:
    plt.axvline(x=f, color='green', linestyle=':', label=f'Intersection at {f:.2f} Hz')
    # Add text label for x-intercept
    plt.text(f,  -1, f'{f:.2f} Hz', color='green', ha='center')

# Show the plot
plt.show()