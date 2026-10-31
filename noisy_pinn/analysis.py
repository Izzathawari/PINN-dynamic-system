import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, Button

# ---------------- data ----------------
r = 3.99
x = 0.2
for _ in range(200):
    x = r * x * (1 - x)
xc = []
for _ in range(500):
    x = r * x * (1 - x)
    xc.append(x)
xc = np.array(xc)
sx = xc.std()

rng = np.random.default_rng(0)
unit_noise = rng.normal(0, 1, len(xc))      # fixed N(0,1) draw, scaled by eta

# ---------------- EKF ----------------
def run_ekf(z, R, Q, P0, second=False):
    n = len(z)
    xh = np.zeros(n)
    xh[0] = z[0]
    P = P0
    for k in range(1, n):
        xp0 = xh[k - 1]
        F = r * (1 - 2 * xp0)
        if second:
            x_pred = r * xp0 * (1 - xp0) - r * P
            P_pred = F**2 * P + 2 * r**2 * P**2 + Q
        else:
            x_pred = r * xp0 * (1 - xp0)
            P_pred = F**2 * P + Q
        g = P_pred / (P_pred + R)
        xh[k] = x_pred + g * (z[k] - x_pred)
        P = (1 - g) * P_pred
    return xh

rmse = lambda a, b: np.sqrt(np.mean((a - b) ** 2))

# ---------------- figure ----------------
N_SHOW = 100
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
fig.subplots_adjust(bottom=0.33, wspace=0.25)

l_true, = ax1.plot([], [], "k-", lw=1.5, label="true")
l_noisy, = ax1.plot([], [], "g.", ms=5, label="noisy")
l_est, = ax1.plot([], [], "r-", lw=1.2, label="EKF")
ax1.set_xlim(0, N_SHOW)
ax1.set_ylim(-0.3, 1.3)
ax1.set_xlabel("k")
ax1.set_ylabel("x")
ax1.legend(loc="upper right")
ax1.grid(alpha=0.3)

m_noisy, = ax2.plot([], [], "g.", ms=4, alpha=0.6, label="noisy")
m_est, = ax2.plot([], [], "r.", ms=4, alpha=0.7, label="EKF")
m_true, = ax2.plot([], [], "k.", ms=3, label="true")
ax2.set_xlim(-0.3, 1.3)
ax2.set_ylim(-0.3, 1.3)
ax2.set_xlabel("x(k)")
ax2.set_ylabel("x(k+1)")
ax2.legend(loc="upper right")
ax2.grid(alpha=0.3)

# ---------------- sliders ----------------
def make_slider(y, label, vmin, vmax, vinit):
    ax = fig.add_axes([0.15, y, 0.55, 0.03])
    return Slider(ax, label, vmin, vmax, valinit=vinit)

s_q = make_slider(0.24, "log10 Q", -8, 0, -3)
s_r = make_slider(0.19, "log10 R (filter)", -4, 0, np.log10((0.3 * sx) ** 2))
s_p = make_slider(0.14, "log10 P0", -6, 1, -2)
s_eta = make_slider(0.09, "eta (noise level)", 0.01, 0.6, 0.3)

ax_chk = fig.add_axes([0.78, 0.12, 0.17, 0.12])
chk = CheckButtons(ax_chk, ["2nd-order predict"], [False])

ax_btn = fig.add_axes([0.78, 0.04, 0.17, 0.06])
btn = Button(ax_btn, "New noise")

# ---------------- update ----------------
def update(_=None):
    eta = s_eta.val
    sigma_noise = eta * sx
    z = xc + sigma_noise * unit_noise

    Q = 10 ** s_q.val
    R = 10 ** s_r.val
    P0 = 10 ** s_p.val
    xh = run_ekf(z, R, Q, P0, second=chk.get_status()[0])

    k = np.arange(N_SHOW)
    l_true.set_data(k, xc[:N_SHOW])
    l_noisy.set_data(k, z[:N_SHOW])
    l_est.set_data(k, xh[:N_SHOW])

    m_true.set_data(xc[:-1], xc[1:])
    m_noisy.set_data(z[:-1], z[1:])
    m_est.set_data(xh[:-1], xh[1:])

    fig.suptitle(
        f"noisy RMSE = {rmse(z, xc):.4f}   |   EKF RMSE = {rmse(xh, xc):.4f}   |   "
        f"true R = {sigma_noise**2:.4f}",
        fontsize=12,
    )
    fig.canvas.draw_idle()

def new_noise(_):
    global unit_noise
    unit_noise = np.random.default_rng().normal(0, 1, len(xc))
    update()

for s in (s_q, s_r, s_p, s_eta):
    s.on_changed(update)
chk.on_clicked(update)
btn.on_clicked(new_noise)

update()
print(f"I am here")
plt.show()