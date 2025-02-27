import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq
import folium

# Ladataan data
accel_data = pd.read_csv("https://raw.githubusercontent.com/oona123/streamlit_folium_project/main/Linear%20Accelerometer.csv?raw=true")
gps_data = pd.read_csv("https://raw.githubusercontent.com/oona123/streamlit_folium_project/main/Location.csv?raw=true")

st.title("Liikunnan analyysi ja visualisointi")
st.write("Tässä projektissa analysoimme kävelyliikettä kiihtyvyys- ja GPS-datan avulla.")

# Funktio matalapäästösuodattimen luomiseksi
def butter_lowpass_filter(data, cutoff=2.5, fs=50, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Askelmäärä suodatetusta kiihtyvyysdatasta
filtered_acc = butter_lowpass_filter(accel_data["Z (m/s^2)"])
peaks, _ = find_peaks(filtered_acc, distance=40)  # Voit säätää distance-parametria
step_count_filtered = len(peaks)

# Fourier-analyysi
N = len(filtered_acc)
T = accel_data["Time (s)"].iloc[1] - accel_data["Time (s)"].iloc[0]
freqs = fftfreq(N, T)
fft_values = fft(filtered_acc)

valid_range = (freqs > 0.5) & (freqs < 3)
peak_freq = freqs[valid_range][abs(fft_values[valid_range]).argmax()]
step_count_fourier = peak_freq * (accel_data["Time (s)"].iloc[-1] - accel_data["Time (s)"].iloc[0])

# GPS-matka ja keskinopeus
lat_rad = np.radians(gps_data["Latitude (°)"])
lon_rad = np.radians(gps_data["Longitude (°)"])
R = 6371000  # Maapallon säde metreinä
distances = R * np.sqrt((lat_rad.diff())**2 + (np.cos(lat_rad) * lon_rad.diff())**2)
total_distance = distances.sum()
time_elapsed = gps_data["Time (s)"].iloc[-1] - gps_data["Time (s)"].iloc[0]
mean_speed = total_distance / time_elapsed
stride_length = total_distance / len(peaks)

# Tulostetaan lasketut tiedot
st.write("#### Lasketut tiedot:")
st.write(f"- **Askelmäärä suodatetusta kiihtyvyysdatasta**: {step_count_filtered} askelta")
st.write(f"- **Askelmäärä Fourier-analyysin perusteella**: {step_count_fourier:.0f} askelta")
st.write(f"- **Keskinopeus**: {mean_speed:.2f} m/s")
st.write(f"- **Kuljettu matka**: {total_distance:.2f} m")
st.write(f"- **Askelpituus**: {stride_length:.2f} m")

# Suodatettu kiihtyvyysdata
st.write("#### Suodatettu kiihtyvyysdata:")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(accel_data["Time (s)"], filtered_acc, label="Suodatettu Z-kiihtyvyys", linewidth=1.5)
ax.set_xlabel("Aika (s)")
ax.set_ylabel("Kiihtyvyys (m/s²)")
ax.set_xlim(10, 60)
ax.legend()
ax.grid(axis='y')
st.pyplot(fig)

# Fourier-analyysi
st.write("#### Tehospektri Fourier-analyysillä:")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(freqs[:N // 2], abs(fft_values[:N // 2]), label="Tehospektri")
ax.set_xlabel("Taajuus (Hz)")
ax.set_ylabel("Teho")
ax.grid(axis='y')
ax.legend()
st.pyplot(fig)

# Kartta ja kuljettu reitti
st.write("#### Reitti kartalla:")
route = list(zip(gps_data["Latitude (°)"], gps_data["Longitude (°)"]))
m = folium.Map(location=[gps_data["Latitude (°)"].mean(), gps_data["Longitude (°)"].mean()], zoom_start=16)
folium.PolyLine(route, color="blue", weight=3, opacity=1).add_to(m)
st.components.v1.html(m._repr_html_(), height=500)