# State-transition autocorrelation-aware null analysis

## Design

The analysis used exactly the 288 monthly EWBI and NIRv values per land type stored for final Figure 9. For each land type, 500 independent EWBI/NIRv IAAFT surrogate pairs were generated with fixed seed 20261013. IAAFT rank-remaps each surrogate to the observed marginal distribution while iteratively matching the Fourier amplitude spectrum, thereby approximately preserving temporal autocorrelation/power. EWBI and NIRv were randomized independently under the null.

The original Figure-9 EWBI and NIRv median thresholds were held fixed; thresholds were not re-estimated for each surrogate. The empirical primary p-value is upper-tailed: the fraction of surrogate persistence probabilities at least as large as observed, with the plus-one correction. A two-sided empirical value is also supplied.

## Results

4 of 16 land-type/state persistence probabilities exceeded the autocorrelation-preserving null at upper-tail p < 0.05. The CSV contains observed persistence, null median, 2.5–97.5% interval, and empirical p-values for every state.

## Dependence-preservation diagnostic

```text
                 ewbi_lag1  nirv_lag1  surrogate_ewbi_lag1_median  surrogate_nirv_lag1_median  ewbi_spectral_error_median  nirv_spectral_error_median
land_type                                                                                                                                            
Grasslands          0.6982     0.7923                      0.6963                      0.7874                      0.0127                      0.0188
Natural Forests     0.6322     0.7343                      0.6303                      0.7292                      0.0126                      0.0129
Savannas            0.6876     0.7959                      0.6857                      0.7886                      0.0131                      0.0129
Shrublands          0.6499     0.8874                      0.6474                      0.8799                      0.0117                      0.0210
```

## Interpretation limit

This test asks whether diagonal persistence exceeds what is expected from independently generated autocorrelated EWBI and NIRv series with the observed marginal distributions. It does not validate a Markov mechanism or establish ecosystem-state dynamics.

Exact input: `C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Drought Paper\AUDIT_TEMP\REPO_REVIEWER_READY\results\states\main_state_classification_deseasonalized.csv`
