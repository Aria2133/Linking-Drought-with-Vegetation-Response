import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from datetime import datetime, timedelta
import glob
import os
from scipy import stats
import matplotlib.dates as mdates
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# تنظیمات فونت برای نمایش فارسی
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'Tahoma']
plt.rcParams['axes.unicode_minus'] = False


class ClimateIndexAnalyzer:
    def __init__(self, spei_dir=None, grace_dir=None, gosif_dir=None):
        """
        کلاس تحلیل شاخص‌های اقلیمی-زیستی

        Parameters:
        -----------
        spei_dir : str
            مسیر پوشه فایل‌های SPEI
        grace_dir : str
            مسیر پوشه فایل‌های GRACE
        gosif_dir : str
            مسیر پوشه فایل‌های GOSIF
        """
        # مسیرهای پیش‌فرض
        if spei_dir is None:
            self.spei_dir = Path(
                r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\FINAL GOSIF AND GRACE AND SPEI\SPEI_tif_Cleaned")
        else:
            self.spei_dir = Path(spei_dir)

        if grace_dir is None:
            self.grace_dir = Path(
                r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\FINAL GOSIF AND GRACE AND SPEI\GRACE_on_SPEI_Grid")
        else:
            self.grace_dir = Path(grace_dir)

        if gosif_dir is None:
            self.gosif_dir = Path(
                r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\FINAL GOSIF AND GRACE AND SPEI\GOSIF_on_SPEI_Grid")
        else:
            self.gosif_dir = Path(gosif_dir)

        self.spei_data = {}
        self.grace_data = {}
        self.gosif_data = {}

    def read_geotiff_files(self):
        """خواندن فایل‌های GeoTIFF و استخراج داده‌ها"""

        print("در حال خواندن فایل‌های SPEI...")
        self._read_spei_files()

        print("در حال خواندن فایل‌های GRACE...")
        self._read_grace_files()

        print("در حال خواندن فایل‌های GOSIF...")
        self._read_gosif_files()

    def _read_spei_files(self):
        """خواندن فایل‌های SPEI"""
        spei_pattern = self.spei_dir / "SPEI_*_*_cleaned.tif"
        spei_files = glob.glob(str(spei_pattern))

        print(f"تعداد فایل‌های SPEI پیدا شده: {len(spei_files)}")

        for file_path in spei_files:
            filename = os.path.basename(file_path)
            try:
                # استخراج سال و ماه از نام فایل
                parts = filename.split('_')
                year = int(parts[1])
                month = int(parts[2])

                # خواندن فایل GeoTIFF
                with rasterio.open(file_path) as src:
                    data = src.read(1)  # خواندن باند اول
                    # محاسبه میانگین برای کل منطقه (حذف NoData)
                    data_clean = data[~np.isnan(data)]
                    if len(data_clean) > 0:
                        self.spei_data[(year, month)] = np.mean(data_clean)
                        print(f"  ✓ SPEI سال {year}, ماه {month}: {np.mean(data_clean):.4f}")
                    else:
                        print(f"  ✗ فایل {filename} داده معتبر ندارد")

            except Exception as e:
                print(f"  ✗ خطا در پردازش {filename}: {str(e)}")
                continue

    def _read_grace_files(self):
        """خواندن فایل‌های GRACE"""
        grace_pattern = self.grace_dir / "GRACE_TWS_Tonzi_*_*_Tonzi_Regridded_Tonzi.tif"
        grace_files = glob.glob(str(grace_pattern))

        print(f"تعداد فایل‌های GRACE پیدا شده: {len(grace_files)}")

        for file_path in grace_files:
            filename = os.path.basename(file_path)
            print(f"در حال پردازش: {filename}")

            try:
                # استخراج سال و ماه از نام فایل
                parts = filename.split('_')
                year = int(parts[3])

                # پردازش ماه با حذف کاراکترهای اضافی
                month_str = parts[4]
                # حذف هر چیزی بعد از عدد (مثل (1) یا حروف)
                month_clean = ""
                for char in month_str:
                    if char.isdigit():
                        month_clean += char
                    else:
                        break

                if month_clean:
                    month = int(month_clean)
                    if 1 <= month <= 12:  # چک کردن معتبر بودن ماه
                        # خواندن فایل GeoTIFF
                        with rasterio.open(file_path) as src:
                            data = src.read(1)
                            # محاسبه میانگین برای کل منطقه
                            data_clean = data[~np.isnan(data)]
                            if len(data_clean) > 0:
                                self.grace_data[(year, month)] = np.mean(data_clean)
                                print(f"  ✓ سال {year}, ماه {month}: {np.mean(data_clean):.4f}")
                            else:
                                print(f"  ✗ فایل {filename} داده معتبر ندارد")
                    else:
                        print(f"  ✗ ماه نامعتبر در {filename}: {month}")
                else:
                    print(f"  ✗ نمی‌توان ماه را از {filename} استخراج کرد")

            except Exception as e:
                print(f"  ✗ خطا در پردازش {filename}: {str(e)}")
                continue

    def _read_gosif_files(self):
        """خواندن فایل‌های GOSIF"""
        gosif_pattern = self.gosif_dir / "GOSIF_*_Enhanced_Tonzi.tif"
        gosif_files = glob.glob(str(gosif_pattern))

        print(f"تعداد فایل‌های GOSIF پیدا شده: {len(gosif_files)}")

        for file_path in gosif_files:
            filename = os.path.basename(file_path)
            try:
                # استخراج سال و ماه از نام فایل
                date_str = filename.split('_')[1]
                year = int(date_str[:4])
                month = int(date_str[4:6])

                # خواندن فایل GeoTIFF
                with rasterio.open(file_path) as src:
                    data = src.read(1)
                    # محاسبه میانگین برای کل منطقه
                    data_clean = data[~np.isnan(data)]
                    if len(data_clean) > 0:
                        self.gosif_data[(year, month)] = np.mean(data_clean)
                        print(f"  ✓ GOSIF سال {year}, ماه {month}: {np.mean(data_clean):.4f}")
                    else:
                        print(f"  ✗ فایل {filename} داده معتبر ندارد")

            except Exception as e:
                print(f"  ✗ خطا در پردازش {filename}: {str(e)}")
                continue

    def fill_missing_grace_data(self):
        """پر کردن داده‌های گمشده GRACE با میانگین وزن‌دار"""

        print("در حال پر کردن داده‌های گمشده GRACE...")

        # ایجاد فهرست کامل ماه‌ها از 2002 تا 2021
        full_dates = []
        for year in range(2002, 2022):
            for month in range(1, 13):
                full_dates.append((year, month))

        # پیدا کردن ماه‌های گمشده
        missing_dates = [date for date in full_dates if date not in self.grace_data]

        for missing_year, missing_month in missing_dates:
            # پیدا کردن تمام داده‌های موجود برای همان ماه تقویمی
            same_month_data = {}
            for (year, month), value in self.grace_data.items():
                if month == missing_month:
                    same_month_data[year] = value

            if len(same_month_data) > 0:
                # محاسبه میانگین وزن‌دار (سال‌های نزدیک‌تر وزن بیشتر)
                weighted_sum = 0
                weight_sum = 0

                for year, value in same_month_data.items():
                    # وزن معکوس فاصله زمانی
                    distance = abs(year - missing_year)
                    weight = 1 / (distance + 1)  # +1 برای جلوگیری از تقسیم بر صفر
                    weighted_sum += value * weight
                    weight_sum += weight

                # محاسبه میانگین وزن‌دار
                self.grace_data[(missing_year, missing_month)] = weighted_sum / weight_sum

        print(f"تعداد ماه‌های پر شده: {len(missing_dates)}")

    def create_time_series_dataframe(self):
        """ایجاد DataFrame برای تحلیل سری زمانی"""

        # ایجاد فهرست کامل تاریخ‌ها
        all_dates = []
        for year in range(2001, 2022):
            for month in range(1, 13):
                all_dates.append(datetime(year, month, 1))

        # ایجاد DataFrame
        df = pd.DataFrame(index=all_dates, columns=['SPEI', 'GRACE', 'GOSIF'],dtype=float)


        # پر کردن داده‌های SPEI
        for (year, month), value in self.spei_data.items():
            date = datetime(year, month, 1)
            if date in df.index:
                df.loc[date, 'SPEI'] = value

        # پر کردن داده‌های GRACE (از 2002)
        for (year, month), value in self.grace_data.items():
            date = datetime(year, month, 1)
            if date in df.index:
                df.loc[date, 'GRACE'] = value

        # پر کردن داده‌های GOSIF
        for (year, month), value in self.gosif_data.items():
            date = datetime(year, month, 1)
            if date in df.index:
                df.loc[date, 'GOSIF'] = value

        return df

    def normalize_to_zscore(self, df):
        """نرمال‌سازی داده‌ها به Z-score به‌صورت ایمن با حفظ dtype عددی"""
        df_numeric = df.apply(pd.to_numeric, errors="coerce")  # اطمینان از float
        df_z = (df_numeric - df_numeric.mean()) / df_numeric.std()
        return df_z

    def create_comparison_plot(self, df_normalized):
        """ایجاد گراف مقایسه‌ای سه شاخص"""

        # Plot settings
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        fig.suptitle('Temporal Trends of Climate–Eco Indicators (2001–2021)',
                     fontsize=16, fontweight='bold', y=0.98)

        # Colors and labels
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        labels = ['SPEI (Drought Index)', 'GRACE TWS (Water Storage)', 'GOSIF (Photosynthesis)']
        columns = ['SPEI', 'GRACE', 'GOSIF']

        # رسم هر شاخص
        for i, (ax, color, label, column) in enumerate(zip(axes, colors, labels, columns)):
            # فیلتر کردن داده‌ها
            data = df_normalized[column].dropna()

            if len(data) > 0:
                # رسم خط
                ax.plot(data.index, data.values, color=color, linewidth=2, alpha=0.8)

                # پر کردن ناحیه مثبت و منفی
                ax.fill_between(data.index, data.values, 0,
                                where=(data.values >= 0), color=color, alpha=0.3)
                ax.fill_between(data.index, data.values, 0,
                                where=(data.values < 0), color=color, alpha=0.3)

                # خط صفر
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)

                # تنظیمات محور
                ax.set_ylabel(f'{label}\n(Z-score)', fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_xlim(datetime(2001, 1, 1), datetime(2021, 12, 31))

                # محدوده y
                y_max = max(abs(data.min()), abs(data.max())) * 1.1
                ax.set_ylim(-y_max, y_max)

                # تنظیمات محور x
                if i == 2:  # آخرین پنل
                    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
                    ax.xaxis.set_major_locator(mdates.YearLocator(2))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                else:
                    ax.set_xticklabels([])

                # آمار توصیفی
                mean_val = data.mean()
                std_val = data.std()
                ax.text(0.02, 0.95, f'Mean: {mean_val:.2f}\nStd Dev: {std_val:.2f}',
                        transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # تنظیمات نهایی
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)

        return fig

    def calculate_correlations(self, df_normalized):
        """محاسبه همبستگی بین شاخص‌ها"""

        print("\n=== Correlation Analysis ===")

        # Pearson correlation
        corr_matrix = df_normalized.corr()
        print("Pearson Correlation Matrix:")
        print(corr_matrix.round(3))

        # Lagged correlation analysis
        print("\n=== Lagged Correlation Analysis ===")

        # بررسی تاخیر SPEI -> GRACE
        max_lag = 6
        lag_corrs_spei_grace = []

        for lag in range(max_lag + 1):
            spei_data = df_normalized['SPEI'].dropna()
            grace_data = df_normalized['GRACE'].shift(-lag).dropna()

            # Find common dates between the two series
            common_dates = spei_data.index.intersection(grace_data.index)
            if len(common_dates) > 10:
                corr = spei_data.loc[common_dates].corr(grace_data.loc[common_dates])
                lag_corrs_spei_grace.append((lag, corr))
                print(f"SPEI -> GRACE (lag = {lag} months): {corr:.3f}")

        # بررسی تاخیر GRACE -> GOSIF
        lag_corrs_grace_gosif = []

        for lag in range(max_lag + 1):
            grace_data = df_normalized['GRACE'].dropna()
            gosif_data = df_normalized['GOSIF'].shift(-lag).dropna()

            # Find common dates between GRACE and GOSIF
            common_dates = grace_data.index.intersection(gosif_data.index)
            if len(common_dates) > 10:
                corr = grace_data.loc[common_dates].corr(gosif_data.loc[common_dates])
                lag_corrs_grace_gosif.append((lag, corr))
                print(f"GRACE -> GOSIF (lag = {lag} months): {corr:.3f}")

        return corr_matrix, lag_corrs_spei_grace, lag_corrs_grace_gosif

    def analyze_trends(self, df_normalized):
        """تحلیل روند زمانی شاخص‌ها"""

        print("\n=== Time Series Trend Analysis ===")

        for column in df_normalized.columns:
            data = df_normalized[column].dropna()
            if len(data) > 0:
                # تبدیل تاریخ به عدد برای رگرسیون
                x = np.arange(len(data))
                y = data.values

                # رگرسیون خطی
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

                # محاسبه روند سالانه
                annual_trend = slope * 12  # 12 ماه

                print(f"{column}:")
                print(f"  Annual trend: {annual_trend:.4f} Z-score/year")
                print(f"  R²: {r_value ** 2:.3f}")
                print(f"  P-value: {p_value:.3f}")
                print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
                print()

    def run_complete_analysis(self):
        """اجرای تحلیل کامل"""

        print("=== Starting Analysis of Climate–Eco Indicators ===")


        # 1. خواندن فایل‌ها
        self.read_geotiff_files()

        # 2. پر کردن داده‌های گمشده GRACE
        self.fill_missing_grace_data()

        # 3. ایجاد DataFrame
        df = self.create_time_series_dataframe()

        # 4. نرمال‌سازی
        df_normalized = self.normalize_to_zscore(df)

        # 5. ایجاد گراف
        fig = self.create_comparison_plot(df_normalized)

        # 6. تحلیل همبستگی
        corr_matrix, lag_corrs_spei_grace, lag_corrs_grace_gosif = self.calculate_correlations(df_normalized)

        # 7. تحلیل روند
        self.analyze_trends(df_normalized)

        # 8. نمایش آمار کلی
        print("\n=== General Data Statistics ===")
        print(f"Number of months in SPEI: {len(self.spei_data)}")
        print(f"Number of months in GRACE: {len(self.grace_data)}")
        print(f"Number of months in GOSIF: {len(self.gosif_data)}")

        return df_normalized, fig, corr_matrix


# نحوه استفاده:
if __name__ == "__main__":
    # ایجاد شیء تحلیلگر با مسیرهای مشخص شده
    analyzer = ClimateIndexAnalyzer()

    # اجرای تحلیل کامل
    df_normalized, fig, corr_matrix = analyzer.run_complete_analysis()

    # نمایش گراف
    plt.show()

    # ذخیره گراف
    fig.savefig('climate_indices_comparison.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')

    # ذخیره داده‌های نرمال‌شده
    df_normalized.to_csv('normalized_climate_indices.csv')

    print("\nAnalysis completed successfully!")
    print("- The graph has been saved as 'climate_indices_comparison.png'")
    print("- The normalized data has been saved as 'normalized_climate_indices.csv'")
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import rasterio
from datetime import datetime
import glob
import os
from pathlib import Path
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
#  Climate‑Eco Indicators – full pipeline (SPEI, GRACE, GOSIF)
#  1) read GeoTIFF files
#  2) fill missing GRACE months by weighted climatology
#  3) build monthly time–series DataFrame (2001‑2021)
#  4) Z‑score normalisation & 3‑panel plot
#  5) correlation, lag‑correlation, trend diagnostics
# -----------------------------------------------------------------------------


class ClimateIndexAnalyzer:
    def __init__(self,
                 spei_dir: str | None = None,
                 grace_dir: str | None = None,
                 gosif_dir: str | None = None):
        """Wrapper for reading & analysing SPEI, GRACE‑TWS and GOSIF time‑series"""
        # default paths (edit if needed)
        self.spei_dir = Path(spei_dir or r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\FINAL GOSIF AND GRACE AND SPEI\SPEI_tif_Cleaned")
        self.grace_dir = Path(grace_dir or r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\FINAL GOSIF AND GRACE AND SPEI\GRACE_on_SPEI_Grid")
        self.gosif_dir = Path(gosif_dir or r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\FINAL GOSIF AND GRACE AND SPEI\GOSIF_on_SPEI_Grid")

        self.spei_data: dict[tuple[int, int], float] = {}
        self.grace_data: dict[tuple[int, int], float] = {}
        self.gosif_data: dict[tuple[int, int], float] = {}

    # -------------------------------------------------------------------------
    # 1. file readers
    # -------------------------------------------------------------------------
    def read_geotiff_files(self):
        print("Reading SPEI files …")
        self._read_spei_files()
        print("Reading GRACE files …")
        self._read_grace_files()
        print("Reading GOSIF files …")
        self._read_gosif_files()

    def _read_spei_files(self):
        spei_files = glob.glob(str(self.spei_dir / "SPEI_*_*_cleaned.tif"))
        print(f"  {len(spei_files)} SPEI rasters found")
        for fp in spei_files:
            fn = os.path.basename(fp)
            try:
                year, month = map(int, fn.split("_")[1:3])
                with rasterio.open(fp) as src:
                    arr = src.read(1).astype(float)
                valid = arr[~np.isnan(arr)]
                if valid.size:
                    self.spei_data[(year, month)] = float(valid.mean())
            except Exception as exc:
                print(f"    !! could not parse {fn}: {exc}")

    def _read_grace_files(self):
        pattern = self.grace_dir / "GRACE_TWS_Tonzi_*_*_Tonzi_Regridded_Tonzi.tif"
        grace_files = glob.glob(str(pattern))
        print(f"  {len(grace_files)} GRACE rasters found")
        for fp in grace_files:
            fn = os.path.basename(fp)
            try:
                parts = fn.split("_")
                year = int(parts[3])
                # strip any trailing chars from month segment
                month = int(''.join(ch for ch in parts[4] if ch.isdigit()))
                if not 1 <= month <= 12:
                    continue
                with rasterio.open(fp) as src:
                    arr = src.read(1).astype(float)
                valid = arr[~np.isnan(arr)]
                if valid.size:
                    self.grace_data[(year, month)] = float(valid.mean())
            except Exception as exc:
                print(f"    !! could not parse {fn}: {exc}")

    def _read_gosif_files(self):
        pattern = self.gosif_dir / "GOSIF_*_Enhanced_Tonzi.tif"
        gosif_files = glob.glob(str(pattern))
        print(f"  {len(gosif_files)} GOSIF rasters found")
        for fp in gosif_files:
            fn = os.path.basename(fp)
            try:
                date_str = fn.split("_")[1]
                year = int(date_str[:4])
                month = int(date_str[4:6])
                with rasterio.open(fp) as src:
                    arr = src.read(1).astype(float)
                valid = arr[~np.isnan(arr)]
                if valid.size:
                    self.gosif_data[(year, month)] = float(valid.mean())
            except Exception as exc:
                print(f"    !! could not parse {fn}: {exc}")

    # -------------------------------------------------------------------------
    # 2. gap‑filling GRACE (weighted climatology)
    # -------------------------------------------------------------------------
    def fill_missing_grace(self):
        print("Filling missing GRACE months …")
        full = [(y, m) for y in range(2002, 2022) for m in range(1, 13)]
        miss = [d for d in full if d not in self.grace_data]
        for yr, mo in miss:
            clim_vals = {y: v for (y, m), v in self.grace_data.items() if m == mo}
            if not clim_vals:
                continue
            w_sum = 0.0
            v_sum = 0.0
            for y, v in clim_vals.items():
                w = 1 / (abs(y - yr) + 1)  # closer years ⇒ higher weight
                w_sum += w
                v_sum += v * w
            self.grace_data[(yr, mo)] = v_sum / w_sum
        print(f"  {len(miss)} months infilled")

    # -------------------------------------------------------------------------
    # 3. build DataFrame
    # -------------------------------------------------------------------------
    def build_dataframe(self):
        dates = [datetime(y, m, 1) for y in range(2001, 2022) for m in range(1, 13)]
        df = pd.DataFrame(index=dates, columns=["SPEI", "GRACE", "GOSIF"], dtype=float)
        for (y, m), v in self.spei_data.items():
            df.loc[datetime(y, m, 1), "SPEI"] = v
        for (y, m), v in self.grace_data.items():
            df.loc[datetime(y, m, 1), "GRACE"] = v
        for (y, m), v in self.gosif_data.items():
            df.loc[datetime(y, m, 1), "GOSIF"] = v
        return df

    # -------------------------------------------------------------------------
    # 4. normalise (column‑wise Z‑score) & plot
    # -------------------------------------------------------------------------
    @staticmethod
    def zscore(df: pd.DataFrame) -> pd.DataFrame:
        df_z = df.copy()
        for col in df.columns:
            series = df[col].dropna()
            if series.empty:
                continue
            df_z[col] = (df[col] - series.mean()) / series.std()
        return df_z

    @staticmethod
    def plot_triple(df_z: pd.DataFrame):
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        fig.suptitle("Temporal Trends of Climate‑Eco Indicators (2001‑2021)",
                     fontsize=16, weight="bold")
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        labels = ["SPEI (Drought Index)", "GRACE TWS (Water Storage)",
                  "GOSIF (Photosynthesis)"]
        for ax, col, lbl, c in zip(axes, df_z.columns, labels, colors):
            s = df_z[col].dropna()
            ax.plot(s.index, s.values, color=c, lw=2)
            ax.fill_between(s.index, s.values, 0, where=s.values >= 0, alpha=.3, color=c)
            ax.fill_between(s.index, s.values, 0, where=s.values < 0, alpha=.3, color=c)
            ax.axhline(0, color="black", ls="--", lw=1)
            ax.set_ylabel(f"{lbl}\n(Z‑score)", weight="bold")
            y_abs = max(abs(s.min()), abs(s.max())) * 1.1
            ax.set_ylim(-y_abs, y_abs)
            ax.grid(alpha=.3)
        axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axes[-1].set_xlabel("Time (year)", weight="bold")
        plt.tight_layout(rect=(0, 0, 1, .96))
        fig.savefig("climate_indices_comparison.png", dpi=300)
        return fig

    # -------------------------------------------------------------------------
    # 5. correlation diagnostics
    # -------------------------------------------------------------------------
    @staticmethod
    def correlations(df_z: pd.DataFrame):
        print("\n=== Correlation Analysis ===")
        print(df_z.corr().round(3))

    @staticmethod
    def trends(df_z: pd.DataFrame):
        print("\n=== Trend Analysis (linear) ===")
        for col in df_z.columns:
            s = df_z[col].dropna()
            if s.empty:
                continue
            x = np.arange(len(s))
            slope, intercept, r, p, _ = stats.linregress(x, s.values)
            print(f"{col:6s} | annual trend: {slope * 12:+.4f} Z/yr  |  R²={r*r:.3f}  |  p={p:.3g}")

    # -------------------------------------------------------------------------
    # 6. lag‑correlation plot (new method)
    # -------------------------------------------------------------------------
    def cross_correlation_lagplot(self,
                                  df_z: pd.DataFrame,
                                  max_lag: int = 12,
                                  min_overlap: int = 24,
                                  save_path: str = "lag_correlation_plot.png"):
        lags = np.arange(-max_lag, max_lag + 1)
        corr_spei, corr_grace = [], []
        for l in lags:
            # SPEI → GOSIF
            c1 = df_z["SPEI"]
            c2 = df_z["GOSIF"].shift(-l)
            idx = c1.index.intersection(c2.dropna().index)
            corr_spei.append(c1.loc[idx].corr(c2.loc[idx]) if len(idx) >= min_overlap else np.nan)
            # GRACE → GOSIF
            c1g = df_z["GRACE"]
            c2g = df_z["GOSIF"].shift(-l)
            idxg = c1g.index.intersection(c2g.dropna().index)
            corr_grace.append(c1g.loc[idxg].corr(c2g.loc[idxg]) if len(idxg) >= min_overlap else np.nan)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(lags, corr_spei, marker="o", label="SPEI → GOSIF")
        ax.plot(lags, corr_grace, marker="s", label="GRACE TWS → GOSIF")
        ax.axhline(0, color="gray", ls="--")
        ax.axvline(0, color="black")
        if not np.all(np.isnan(corr_spei)):
            i = np.nanargmax(np.abs(corr_spei))
            ax.scatter(lags[i], corr_spei[i], color="blue", s=100, zorder=5)
        if not np.all(np.isnan(corr_grace)):
            j = np.nanargmax(np.abs(corr_grace))
            ax.scatter(lags[j], corr_grace[j], color="orange", s=100, zorder=5)
        ax.set_xlabel("Lag (months) – positive = climate leads", weight="bold")
        ax.set_ylabel("Pearson r", weight="bold")
        ax.set_title("Lagged response of vegetation (GOSIF)", weight="bold")
        ax.grid(alpha=.3)
        ax.legend(frameon=False)
        ax.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
        plt.tight_layout()
        fig.savefig(save_path, dpi=300)
        return fig

    # -------------------------------------------------------------------------
    #  Orchestrator
    # -------------------------------------------------------------------------
    def run(self):
        print("=== Climate‑Eco indicator analysis started ===")
        self.read_geotiff_files()
        self.fill_missing_grace()
        df = self.build_dataframe()
        df_z = self.zscore(df)
        fig_triple = self.plot_triple(df_z)
        self.correlations(df_z)
        self.trends(df_z)
        fig_lag = self.cross_correlation_lagplot(df_z)
        print("\n--- done! results saved as PNG files in current folder ---")
        return df_z, fig_triple, fig_lag


# -----------------------------------------------------------------------------
#  Main execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    analyzer = ClimateIndexAnalyzer()
    df_normalized, fig1, fig2 = analyzer.run()
    plt.show()  # show any open figures interactively
