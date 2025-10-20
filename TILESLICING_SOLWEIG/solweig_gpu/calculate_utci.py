# ==================== solweig_gpu/utci_process.py ====================
# UTCI via NumPy with safety clamps + optional per-point debug logging.
# Includes a Torch wrapper (CPU roundtrip) so the rest of the GPU pipeline
# can keep using tensors.

from __future__ import annotations
import numpy as np
import torch

# ---------- 6th-order UTCI polynomial (NumPy) ----------
def utci_polynomial(D_Tmrt, Ta, va, Pa):
    # All inputs are NumPy arrays / scalars (broadcastable).
    Ta2 = Ta * Ta
    Ta3 = Ta2 * Ta
    Ta4 = Ta3 * Ta
    Ta5 = Ta4 * Ta
    Ta6 = Ta5 * Ta

    va2 = va * va
    va3 = va2 * va
    va4 = va3 * va
    va5 = va4 * va
    va6 = va5 * va

    DT  = D_Tmrt
    DT2 = DT * DT
    DT3 = DT2 * DT
    DT4 = DT3 * DT
    DT5 = DT4 * DT
    DT6 = DT5 * DT

    # (coefficients identical to official UTCI approximation)
    return (
        Ta
        + 6.07562052e-01
        - 2.27712343e-02 * Ta
        + 8.06470249e-04 * Ta2
        - 1.54271372e-04 * Ta3
        - 3.24651735e-06 * Ta4
        + 7.32602852e-08 * Ta5
        + 1.35959073e-09 * Ta6
        - 2.25836520e+00 * va
        + 8.80326035e-02  * Ta * va
        + 2.16844454e-03  * Ta2 * va
        - 1.53347087e-05  * Ta3 * va
        - 5.72983704e-07  * Ta4 * va
        - 2.55090145e-09  * Ta5 * va
        - 7.51269505e-01  * va2
        - 4.08350271e-03  * Ta * va2
        - 5.21670675e-05  * Ta2 * va2
        + 1.94544667e-06  * Ta3 * va2
        + 1.14099531e-08  * Ta4 * va2
        + 1.58137256e-01  * va3
        - 6.57263143e-05  * Ta * va3
        + 2.22697524e-07  * Ta2 * va3
        - 4.16117031e-08  * Ta3 * va3
        - 1.27762753e-02  * va4
        + 9.66891875e-06  * Ta * va4
        + 2.52785852e-09  * Ta2 * va4
        + 4.56306672e-04  * va5
        - 1.74202546e-07  * Ta * va5
        - 5.91491269e-06  * va6
        + 3.98374029e-01  * DT
        + 1.83945314e-04  * Ta * DT
        - 1.73754510e-04  * Ta2 * DT
        - 7.60781159e-07  * Ta3 * DT
        + 3.77830287e-08  * Ta4 * DT
        + 5.43079673e-10  * Ta5 * DT
        - 2.00518269e-02  * va * DT
        + 8.92859837e-04  * Ta * va * DT
        + 3.45433048e-06  * Ta2 * va * DT
        - 3.77925774e-07  * Ta3 * va * DT
        - 1.69699377e-09  * Ta4 * va * DT
        + 1.69992415e-04  * va2 * DT
        - 4.99204314e-05  * Ta * va2 * DT
        + 2.47417178e-07  * Ta2 * va2 * DT
        + 1.07596466e-08  * Ta3 * va2 * DT
        + 8.49242932e-05  * va3 * DT
        + 1.35191328e-06  * Ta * va3 * DT
        - 6.21531254e-09  * Ta2 * va3 * DT
        - 4.99410301e-06  * va4 * DT
        - 1.89489258e-08  * Ta * va4 * DT
        + 8.15300114e-08  * va5 * DT
        + 7.55043090e-04  * DT2
        - 5.65095215e-05  * Ta * DT2
        - 4.52166564e-07  * Ta2 * DT2
        + 2.46688878e-08  * Ta3 * DT2
        + 2.42674348e-10  * Ta4 * DT2
        + 1.54547250e-04  * va * DT2
        + 5.24110970e-06  * Ta * va * DT2
        - 8.75874982e-08  * Ta2 * va * DT2
        - 1.50743064e-09  * Ta3 * va * DT2
        - 1.56236307e-05  * va2 * DT2
        - 1.33895614e-07  * Ta * va2 * DT2
        + 2.49709824e-09  * Ta2 * va2 * DT2
        + 6.51711721e-07  * va3 * DT2
        + 1.94960053e-09  * Ta * va3 * DT2
        - 1.00361113e-08  * va4 * DT2
        - 1.21206673e-05  * DT3
        - 2.18203660e-07  * Ta * DT3
        + 7.51269482e-09  * Ta2 * DT3
        + 9.79063848e-11  * Ta3 * DT3
        + 1.25006734e-06  * va * DT3
        - 1.81584736e-09  * Ta * va * DT3
        - 3.52197671e-10  * Ta2 * va * DT3
        - 3.36514630e-08  * va2 * DT3
        + 1.35908359e-10  * Ta * va2 * DT3
        + 4.17032620e-10  * va3 * DT3
        - 1.30369025e-09  * DT4
        + 4.13908461e-10  * Ta * DT4
        + 9.22652254e-12  * Ta2 * DT4
        - 5.08220384e-09  * va * DT4
        - 2.24730961e-11  * Ta * va * DT4
        + 1.17139133e-10  * va2 * DT4
        + 6.62154879e-10  * DT5
        + 4.03863260e-13  * Ta * DT5
        + 1.95087203e-12  * va * DT5
        - 4.73602469e-12  * DT6
        + 5.12733497e+00  * Pa
        - 3.12788561e-01  * Ta * Pa
        - 1.96701861e-02  * Ta2 * Pa
        + 9.99690870e-04  * Ta3 * Pa
        + 9.51738512e-06  * Ta4 * Pa
        - 4.66426341e-07  * Ta5 * Pa
        + 5.48050612e-01  * va * Pa
        - 3.30552823e-03  * Ta * va * Pa
        - 1.64119440e-03  * Ta2 * va * Pa
        - 5.16670694e-06  * Ta3 * va * Pa
        + 9.52692432e-07  * Ta4 * va * Pa
        - 4.29223622e-02  * va2 * Pa
        + 5.00845667e-03  * Ta * va2 * Pa
        + 1.00601257e-06  * Ta2 * va2 * Pa
        - 1.81748644e-06  * Ta3 * va2 * Pa
        - 1.25813502e-03  * va3 * Pa
        - 1.79330391e-04  * Ta * va3 * Pa
        + 2.34994441e-06  * Ta2 * va3 * Pa
        + 1.29735808e-04  * va4 * Pa
        + 1.29064870e-06  * Ta * va4 * Pa
        - 2.28558686e-06  * va5 * Pa
        - 3.69476348e-02  * DT * Pa
        + 1.62325322e-03  * Ta * DT * Pa
        - 3.14279680e-05  * Ta2 * DT * Pa
        + 2.59835559e-06  * Ta3 * DT * Pa
        - 4.77136523e-08  * Ta4 * DT * Pa
        + 8.64203390e-03  * va * DT * Pa
        - 6.87405181e-04  * Ta * va * DT * Pa
        - 9.13863872e-06  * Ta2 * va * DT * Pa
        + 5.15916806e-07  * Ta3 * va * DT * Pa
        - 3.59217476e-05  * va2 * DT * Pa
        + 3.28696511e-05  * Ta * va2 * DT * Pa
        - 7.10542454e-07  * Ta2 * va2 * DT * Pa
        - 1.24382300e-05  * va3 * DT * Pa
        - 7.38584400e-09  * Ta * va3 * DT * Pa
        + 2.20609296e-07  * va4 * DT * Pa
        - 7.32469180e-04  * DT2 * Pa
        - 1.87381964e-05  * Ta * DT2 * Pa
        + 4.80925239e-06  * Ta2 * DT2 * Pa
        - 8.75492040e-08  * Ta3 * DT2 * Pa
        + 2.77862930e-05  * va * DT2 * Pa
        - 5.06004592e-06  * Ta * va * DT2 * Pa
        + 1.14325367e-07  * Ta2 * va * DT2 * Pa
        + 2.53016723e-06  * va2 * DT2 * Pa
        - 1.72857035e-08  * Ta * va2 * DT2 * Pa
        - 3.95079398e-08  * va3 * DT2 * Pa
        - 3.59413173e-07  * DT3 * Pa
        + 7.04388046e-07  * Ta * DT3 * Pa
        - 1.89309167e-08  * Ta2 * DT3 * Pa
        - 4.79768731e-07  * va * DT3 * Pa
        + 7.96079978e-09  * Ta * va * DT3 * Pa
        + 1.62897058e-09  * va2 * DT3 * Pa
        + 3.94367674e-08  * DT4 * Pa
        - 1.18566247e-09  * Ta * DT4 * Pa
        + 3.34678041e-10  * va * DT4 * Pa
        - 1.15606447e-10  * DT5 * Pa
        - 2.80626406e+00  * Pa * Pa
        + 5.48712484e-01  * Ta * Pa * Pa
        - 3.99428410e-03  * Ta2 * Pa * Pa
        - 9.54009191e-04  * Ta3 * Pa * Pa
        + 1.93090978e-05  * Ta4 * Pa * Pa
        - 3.08806365e-01  * va * Pa * Pa
        + 1.16952364e-02  * Ta * va * Pa * Pa
        + 4.95271903e-04  * Ta2 * va * Pa * Pa
        - 1.90710882e-05  * Ta3 * va * Pa * Pa
        + 2.10787756e-03  * va2 * Pa
        - 6.98445738e-04  * Ta * va2 * Pa
        + 2.30109073e-05  * Ta2 * va2 * Pa
        + 4.17856590e-04  * va3 * Pa
        - 1.27043871e-05  * Ta * va3 * Pa
        - 3.04620472e-06  * va4 * Pa
        + 5.14507424e-02  * DT * Pa * Pa
        - 4.32510997e-03  * Ta * DT * Pa * Pa
        + 8.99281156e-05  * Ta2 * DT * Pa * Pa
        - 7.14663943e-07  * Ta3 * DT * Pa * Pa
        - 2.66016305e-04  * va * DT * Pa * Pa
        + 2.63789586e-04  * Ta * va * DT * Pa * Pa
        - 7.01199003e-06  * Ta2 * va * DT * Pa * Pa
        - 1.06823306e-04  * va2 * DT * Pa * Pa
        + 3.61341136e-06  * Ta * va2 * DT * Pa * Pa
        + 2.29748967e-07  * va3 * DT * Pa * Pa
        + 3.04788893e-04  * DT2 * Pa * Pa
        - 6.42070836e-05  * Ta * DT2 * Pa * Pa
        + 1.16257971e-06  * Ta2 * DT2 * Pa * Pa
        + 7.68023384e-06  * va * DT2 * Pa * Pa
        - 5.47446896e-07  * Ta * va * DT2 * Pa * Pa
        - 3.59937910e-08  * va2 * DT2 * Pa * Pa
        - 4.36497725e-06  * DT3 * Pa * Pa
        + 1.68737969e-07  * Ta * DT3 * Pa * Pa
        + 2.67489271e-08  * va * DT3 * Pa * Pa
        + 3.23926897e-09  * DT4 * Pa * Pa
        - 3.53874123e-02  * Pa**3
        - 2.21201190e-01  * Ta * Pa**3
        + 1.55126038e-02  * Ta2 * Pa**3
        - 2.63917279e-04  * Ta3 * Pa**3
        + 4.53433455e-02  * va * Pa**3
        - 4.32943862e-03  * Ta * va * Pa**3
        + 1.45389826e-04  * Ta2 * va * Pa**3
        + 2.17508610e-04  * va2 * Pa**3
        - 6.66724702e-05  * Ta * va2 * Pa**3
        + 3.33217140e-05  * va3 * Pa**3
        - 2.26921615e-03  * DT * Pa**3
        + 3.80261982e-04  * Ta * DT * Pa**3
        - 5.45314314e-09  * Ta2 * DT * Pa**3
        - 7.96355448e-04  * va * DT * Pa**3
        + 2.53458034e-05  * Ta * va * DT * Pa**3
        - 6.31223658e-06  * va2 * DT * Pa**3
        + 3.02122035e-04  * DT2 * Pa**3
        - 4.77403547e-06  * Ta * DT2 * Pa**3
        + 1.73825715e-06  * va * DT2 * Pa**3
        - 4.09087898e-07  * DT3 * Pa**3
        + 6.14155345e-01  * Pa**4
        - 6.16755931e-02  * Ta * Pa**4
        + 1.33374846e-03  * Ta2 * Pa**4
        + 3.55375387e-03  * va * Pa**4
        - 5.13027851e-04  * Ta * va * Pa**4
        + 1.02449757e-04  * va2 * Pa**4
        - 1.48526421e-03  * DT * Pa**4
        - 4.11469183e-05  * Ta * DT * Pa**4
        - 6.80434415e-06  * va * DT * Pa**4
        - 9.77675906e-06  * DT2 * Pa**4
        + 8.82773108e-02  * Pa**5
        - 3.01859306e-03  * Ta * Pa**5
        + 1.04452989e-03  * va * Pa**5
        + 2.47090539e-04  * DT * Pa**5
        + 1.48348065e-03  * Pa**6
    )


# Cache dei coefficienti per pressione di vapore
_VAP_COEFF_CACHE = {}

def _vapour_es_hpa_torch_fast(Ta_v: torch.Tensor) -> torch.Tensor:
    """Saturation vapour pressure (hPa), ottimizzato e senza loop lenti."""
    dev = Ta_v.device
    g = _VAP_COEFF_CACHE.get(dev)
    if g is None:
        g = torch.tensor([
            -2.8365744e3, -6.028076559e3, 1.954263612e1, -2.737830188e-2,
             1.6261698e-5,  7.0229056e-10, -1.8680009e-13,  2.7150305
        ], dtype=torch.float32, device=dev)
        _VAP_COEFF_CACHE[dev] = g

    tk = Ta_v + 273.15
    inv_tk = 1.0 / tk
    tk_m1, tk_m2 = inv_tk, inv_tk * inv_tk
    tk0 = torch.ones_like(tk)
    tk1, tk2, tk3, tk4 = tk, tk*tk, tk*tk*tk, tk*tk*tk*tk

    poly = (g[0]*tk_m2 + g[1]*tk_m1 + g[2]*tk0 +
            g[3]*tk1 + g[4]*tk2 + g[5]*tk3 + g[6]*tk4)
    es = torch.exp(g[7]*torch.log(tk) + poly) * 0.01
    return es


def utci_calculator(Ta, RH, Tmrt, va10m):
    """
    UTCI (°C) via polinomio 6° ordine, versione veloce + debug.
    - Clamp RH [1,100] %, vento ≥0.10 m/s
    - Se UTCI>60 al primo calcolo, stampa debug e ricalcola con vento ≥0.70
    - Clamp finale UTCI [-60,60]
    - Punti invalidi (<= -999) restano -999
    """
    invalid_mask = (Ta <= -999) | (RH <= -999) | (va10m <= -999) | (Tmrt <= -999)
    valid_mask = ~invalid_mask

    UTCI_approx = torch.full_like(Ta, -999, dtype=torch.float32)
    if not torch.any(valid_mask):
        return UTCI_approx

    Ta_v   = Ta[valid_mask].to(torch.float32)
    RH_v   = RH[valid_mask].to(torch.float32)
    Tmrt_v = Tmrt[valid_mask].to(torch.float32)
    va_v   = va10m[valid_mask].to(torch.float32)

    RH_v = torch.clamp(RH_v, 1.0, 100.0)
    va_v = torch.maximum(va_v, torch.tensor(0.10, dtype=va_v.dtype, device=va_v.device))

    es_hpa = _vapour_es_hpa_torch_fast(Ta_v)
    Pa = (es_hpa * (RH_v * 0.01)) * 0.1  # kPa
    D_Tmrt = Tmrt_v - Ta_v

    utci_v = utci_polynomial(D_Tmrt, Ta_v, va_v, Pa).to(torch.float32)

    # DEBUG se UTCI > 60 già al primo pass
    high = utci_v > 60.0
    if torch.any(high):
        idx = torch.nonzero(high, as_tuple=False).view(-1)
        max_print = min(idx.numel(), 10)
        print(f"[UTCI WARN] {idx.numel()} valori con UTCI > 60 dopo primo pass. Mostro i primi {max_print}:")
        for k in range(max_print):
            i = idx[k].item()
            print(f"  -> i={i}: Ta={float(Ta_v[i]):.2f} °C, RH={float(RH_v[i]):.2f} %, "
                  f"Tmrt={float(Tmrt_v[i]):.2f} °C, va={float(va_v[i]):.3f} m/s, "
                  f"Pa={float(Pa[i]):.3f} kPa, D_Tmrt={float(D_Tmrt[i]):.2f} K, "
                  f"UTCI={float(utci_v[i]):.2f} °C")

        # Ricalcola con vento min=0.70
        va_hi = torch.maximum(va_v[high], torch.tensor(0.70, dtype=va_v.dtype, device=va_v.device))
        utci_hi = utci_polynomial(D_Tmrt[high], Ta_v[high], va_hi, Pa[high]).to(torch.float32)
        utci_v[high] = utci_hi

    utci_v = torch.clamp(utci_v, -60.0, 60.0)
    UTCI_approx[valid_mask] = utci_v
    return UTCI_approx
