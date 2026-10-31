import numpy as np
from EKF import EKF
import matplotlib.pyplot as plt

def create_data():
    r = 3.99
    x_current = 0.2
    x_hist = []

    for _ in range(200):  # Burn-in transients
        x_current = r * x_current * (1 - x_current)

    for _ in range(500):  # Collect data
        x_next = r * x_current * (1 - x_current)
        x_hist.append(x_next)
        x_current = x_next

    x_hist = np.array(x_hist) # Convert into numpy

    # --- 2. Apply the Paper's Noise Formula ---
    eta = 0.3  # % noise-to-signal ratio
    sigma_x = np.std(x_hist)  # Standard deviation of the clean signal
    sigma_noise = eta * sigma_x  # Scaled noise standard deviation
    print(f"Standard Deviation Clean Signal {sigma_x}")
    print(f"Standard Deviation Noisy Signal {sigma_noise}")

    # Generate white noise using the paper's formula
    white_noise = np.random.normal(loc=0.0, scale=sigma_noise, size=len(x_hist))

    # Construct noisy time series s(k)
    s_k = x_hist + white_noise

    return x_hist,s_k, r, sigma_noise

def run_EKF():

    x_hist,s_k, r, sigma_noise = create_data()

    f  = lambda x: r * x * (1 - x)
    df = lambda x: r * (1 - 2 * x)

    R_meas = np.array([[sigma_noise**2]])
    Q = np.array([[1e-4]])
    H = np.array([[1.0]])   
    print(f"Sigma Noise {sigma_noise :.4f}")                   # h(x) = x

    ekf = EKF(x0=s_k[0], P0=[[sigma_noise**2]])
    x_est = [s_k[0]]

    for k in range(1, len(s_k)):
        x_prev = ekf.x                          # previous estimate
        F = np.array([[df(x_prev[0])]])         # slope at previous estimate
        ekf.predict(f(x_prev), F, Q)            # f(x_prev) is the true map
        ekf.update(s_k[k], ekf.x, H, R_meas)    # h(x_pred) = x_pred
        ekf.x = np.clip(ekf.x, 0, 1)            # keep inside the map's range
        x_est.append(ekf.x[0])

    x_est = np.array(x_est)
    print("noisy RMSE:   ", np.sqrt(np.mean((s_k - x_hist)**2)))
    print("filtered RMSE:", np.sqrt(np.mean((x_est - x_hist)**2)))
    
    plt.figure(figsize=(7,5))
    plt.plot(x_est[:-1], x_est[1:], ".",color="red" , label="EKF estimation")
    plt.plot(x_hist[:-1], x_hist[1:], ".",color="black", label="True Data" )
    plt.plot(s_k[:-1], s_k[1:],".",color="green", label="Noisy Data" )
    plt.legend()

    plt.figure(figsize=(10,6))
    plt.plot(x_est[:50], "o",linestyle = "-",color="red", label="EKF estimation" )
    plt.plot(x_hist[:50], "o",linestyle = "-",color="black", label="True Data" )
    plt.plot(s_k[:50], "o",linestyle = "-",color="green", label="Noisy Data" )
    plt.legend()
    plt.show()
    return 0

def main ():
    run_EKF()

    return 0

if __name__ == "__main__":
    main()