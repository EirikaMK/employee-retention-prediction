# -*- coding: utf-8 -*-
"""
Employee Intention to Stay Analysis Engine
This module contains all the original analysis logic 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import math
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class EmployeeITSAnalysis:
    """Main analysis class that preserves all original functionality"""
    
    def __init__(self, data_path):
        """Initialize with dataset path"""
        self.data_path = data_path
        self.df = None
        self.df_num = None
        self.numeric_cols = []
        self.categorical_cols = []
        self.q_vars = []
        self.pos_vars = []
        self.js_vars = []
        self.wlb_vars = []
        self.its_vars = []
        
    def load_data(self):
        """1.2. Upload and read the dataset"""
        self.df = pd.read_csv(self.data_path)
        return self.df
    
    def inspect_dataframe(self):
        """1.3. Inspect DataFrame structure and variable types"""
        print("\n=== DataFrame Info ===")
        self.df.info()
        # Convert all columns to 'category' (since all are Likert-scale responses)
        self.df = self.df.astype('category')
        print("\n=== Variable Types ===")
        print(self.df.dtypes)
        return self.df.dtypes
    
    def separate_columns(self):
        """1.4. Separate numeric and categorical columns"""
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        # Exclude timestamp if present from categorical
        if 'Timestamp' in self.categorical_cols:
            self.categorical_cols.remove('Timestamp')
        
        print("\nNumeric Columns:", self.numeric_cols)
        print("Categorical Columns:", self.categorical_cols)
        return self.numeric_cols, self.categorical_cols
    
    def check_missing_values(self):
        """1.5. Check missing data per variable"""
        print("\n=== Missing Values per Column ===")
        missing_values = self.df.isnull().sum()
        print(missing_values)
        return missing_values
    
    def impute_missing_values(self):
        """1.6. Impute missing values"""
        # Numeric: mean imputation
        for col in self.numeric_cols:
            self.df[col].fillna(self.df[col].mean(), inplace=True)
        
        # Categorical: mode imputation
        for col in self.categorical_cols:
            self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        
        print("\n=== Missing Values After Imputation ===")
        missing_after = self.df.isnull().sum()
        print(missing_after)
        return missing_after
    
    def generate_bar_charts(self, save_path=None):
        """1.7: Bar Chart for Each Variable"""
        categorical_cols = self.df.columns  # All columns are ordinal categorical
        charts = []
        
        for col in categorical_cols:
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.countplot(x=col, data=self.df, ax=ax)
            ax.set_title(f"Bar Chart of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            plt.tight_layout()
            
            if save_path:
                plt.savefig(f"{save_path}/bar_chart_{col}.png", dpi=150, bbox_inches='tight')
                charts.append(f"{save_path}/bar_chart_{col}.png")
            plt.close()
        
        return charts
    
    def generate_cross_tables(self):
        """1.8: Cross Table: Q1 to Q8 vs ITS1 to ITS5"""
        cross_tables = {}
        
        for its in [f'ITS{i}' for i in range(1, 6)]:
            cross_tables[its] = {}
            print(f"\n====== {its}: Q1–Q8 Cross Tables ======")
            for q in [f'Q{i}' for i in range(1, 9)]:
                print(f"\nCross Table: {q} vs {its}")
                ctab = pd.crosstab(self.df[q], self.df[its])
                print(ctab)
                cross_tables[its][q] = ctab
        
        return cross_tables
    
    def detect_variable_blocks(self):
        """Auto-detect variable blocks for bivariate analysis"""
        import re
        
        # Reload and clean data
        df_temp = pd.read_csv(self.data_path)
        df_temp.columns = [c.strip() for c in df_temp.columns]
        
        def pick(pattern):
            r = re.compile(pattern)
            return [c for c in df_temp.columns if r.fullmatch(c)]
        
        self.q_vars   = pick(r"Q[1-9]\d?") or [c for c in df_temp.columns if c.upper().startswith("Q")]
        self.pos_vars = pick(r"POS[1-9]\d?") or [c for c in df_temp.columns if c.upper().startswith("POS")]
        self.js_vars  = pick(r"JS[1-9]\d?") or [c for c in df_temp.columns if c.upper().startswith("JS")]
        self.wlb_vars = pick(r"WLB[1-9]\d?") or [c for c in df_temp.columns if c.upper().startswith("WLB")]
        self.its_vars = pick(r"ITS[1-9]\d?") or [c for c in df_temp.columns if c.upper().startswith("ITS")]
        
        # Convert all relevant columns to numeric
        all_vars = list(dict.fromkeys(self.q_vars + self.pos_vars + self.js_vars + self.wlb_vars + self.its_vars))
        self.df_num = df_temp[all_vars].apply(pd.to_numeric, errors="coerce")
        
        return {
            'Q': self.q_vars,
            'POS': self.pos_vars,
            'JS': self.js_vars,
            'WLB': self.wlb_vars,
            'ITS': self.its_vars
        }
    
    def generate_pivot_tables(self):
        """1.9: Bivariate Pivot Tables — Q1 to Q8 vs Key Variables"""
        if self.df_num is None:
            self.detect_variable_blocks()
        
        pivot_tables = {}
        
        for q in self.q_vars:
            print(f"\n==============================")
            print(f"Pivot Tables for {q}")
            print("==============================\n")
            
            pivot_tables[q] = {}
            
            # POS/JS/WLB vs ITS (Mean tables)
            for block_name, vars_list in [("POS", self.pos_vars), ("JS", self.js_vars), ("WLB", self.wlb_vars)]:
                pivot_tables[q][block_name] = {}
                for var in vars_list:
                    pivot_tables[q][block_name][var] = {}
                    for its in self.its_vars:
                        pivot = self.df_num.pivot_table(values=var, index=q, columns=its, aggfunc="mean")
                        print(f"Average {var} by {q} grouped by {its}:")
                        print(pivot)
                        print("\n" + "-"*50 + "\n")
                        pivot_tables[q][block_name][var][its] = pivot
            
            # ITS vs Q (Count tables)
            pivot_tables[q]['ITS_counts'] = {}
            for its in self.its_vars:
                pivot = self.df_num.pivot_table(index=q, columns=its, aggfunc="size", fill_value=0)
                print(f"Count of {its} responses by {q}:")
                print(pivot)
                print("\n" + "-"*50 + "\n")
                pivot_tables[q]['ITS_counts'][its] = pivot
        
        return pivot_tables
    
    def generate_stacked_bar_charts(self, save_path=None):
        """1.10. Stacked Bar Charts (Q1–Q8 vs ITS1-ITS5)"""
        charts = []
        
        for q_col in [f'Q{i}' for i in range(1, 9)]:
            for its_col in [f'ITS{i}' for i in range(1, 6)]:
                # Crosstab (counts)
                ctab = pd.crosstab(self.df[q_col], self.df[its_col])
                
                # Plot stacked bar
                fig, ax = plt.subplots(figsize=(6, 3))
                ctab.plot(kind='bar', stacked=True, ax=ax)
                
                # Title and labels
                ax.set_title(f'Stacked Bar Chart: {q_col} vs {its_col}', fontsize=13)
                ax.set_xlabel(q_col)
                ax.set_ylabel('Number of Responses')
                ax.legend(title=its_col, bbox_to_anchor=(1.05, 1), loc='upper left')
                
                plt.tight_layout()
                
                if save_path:
                    filename = f"{save_path}/stacked_{q_col}_vs_{its_col}.png"
                    plt.savefig(filename, dpi=150, bbox_inches='tight')
                    charts.append(filename)
                plt.close()
        
        return charts
    
    def generate_facet_grids_its_by_q(self, save_path=None):
        """Plot 1: FacetGrid — Distribution of ITS1-ITS5 by Q1 to Q8"""
        charts = []
        
        for its in [f'ITS{i}' for i in range(1, 6)]:
            # Reshape data for plotting (long-form)
            df_melted = self.df.melt(
                id_vars=its,
                value_vars=[f'Q{i}' for i in range(1, 9)],
                var_name='Question',
                value_name='Response'
            )
            
            # Convert to string for proper hue grouping
            df_melted[its] = df_melted[its].astype(str)
            df_melted['Response'] = df_melted['Response'].astype(str)
            
            # Plot FacetGrid
            g = sns.catplot(
                data=df_melted, kind='count',
                x='Response', hue=its,
                col='Question', col_wrap=4,
                height=4, aspect=1.2,
                palette='Set2'
            )
            
            # Titles and axis labels
            g.fig.subplots_adjust(top=0.9)
            g.fig.suptitle(f'Distribution of {its} by Q1 to Q8', fontsize=16)
            g.set_axis_labels("Response (Likert Scale)", "Number of Employees")
            g.set_titles("{col_name}")
            
            if save_path:
                filename = f"{save_path}/facetgrid_{its}_by_Q.png"
                g.savefig(filename, dpi=150, bbox_inches='tight')
                charts.append(filename)
            plt.close()
        
        return charts
    
    def generate_facet_grids_its_by_pos(self, save_path=None):
        """Plot 2: FacetGrid — Distribution of ITS1-ITS5 by POS1 to POS8"""
        charts = []
        
        # Clean column names
        self.df.columns = [c.strip() for c in self.df.columns]
        
        its_list = [f"ITS{i}" for i in range(1, 6)]
        pos_list = [f"POS{i}" for i in range(1, 9)]
        likert_order = ['1','2','3','4','5']
        
        for its in its_list:
            # Long-form reshape
            df_melt = self.df.melt(
                id_vars=its,
                value_vars=pos_list,
                var_name="POS_Item",
                value_name="Response"
            )
            
            # Treat as categorical strings
            df_melt[its] = pd.Categorical(df_melt[its].astype(str), categories=likert_order, ordered=True)
            df_melt["Response"] = pd.Categorical(df_melt["Response"].astype(str), categories=likert_order, ordered=True)
            
            # Faceted count plots
            g = sns.catplot(
                data=df_melt, kind="count",
                x="Response", hue=its,
                col="POS_Item", col_wrap=3,
                height=4, aspect=1.2,
                palette="Set2",
                order=likert_order
            )
            
            # Titles & labels
            g.fig.subplots_adjust(top=0.9)
            g.fig.suptitle(f"Distribution of {its} by POS1–POS8", fontsize=16)
            g.set_axis_labels("Response (Likert Scale)", "Count")
            g.set_titles("{col_name}")
            g.add_legend(title=its)
            
            if save_path:
                filename = f"{save_path}/facetgrid_{its}_by_POS.png"
                g.savefig(filename, dpi=150, bbox_inches='tight')
                charts.append(filename)
            plt.close()
        
        return charts
    
    def generate_facet_grids_its_by_js(self, save_path=None):
        """Plot 3: FacetGrid — Distribution of ITS1-ITS5 by JS1 to JS5"""
        charts = []
        
        self.df.columns = [c.strip() for c in self.df.columns]
        
        its_list = [f"ITS{i}" for i in range(1, 6)]
        js_list = [f"JS{i}" for i in range(1, 6)]
        likert_order = ['1', '2', '3', '4', '5']
        
        for its in its_list:
            # Melt to long format
            df_js = self.df.melt(
                id_vars=its,
                value_vars=js_list,
                var_name='JS_Question',
                value_name='Response'
            )
            
            # Ensure categorical strings with fixed order
            df_js[its] = pd.Categorical(df_js[its].astype(str), categories=likert_order, ordered=True)
            df_js['Response'] = pd.Categorical(df_js['Response'].astype(str), categories=likert_order, ordered=True)
            
            # FacetGrid
            g = sns.catplot(
                data=df_js, kind='count',
                x='Response', hue=its,
                col='JS_Question', col_wrap=3,
                height=4, aspect=1.2,
                palette='Set2',
                order=likert_order
            )
            
            g.fig.subplots_adjust(top=0.9)
            g.fig.suptitle(f'Distribution of {its} by JS1–JS5', fontsize=16)
            g.set_axis_labels('Response (Likert Scale)', 'Count')
            g.set_titles('{col_name}')
            g.add_legend(title=its)
            
            if save_path:
                filename = f"{save_path}/facetgrid_{its}_by_JS.png"
                g.savefig(filename, dpi=150, bbox_inches='tight')
                charts.append(filename)
            plt.close()
        
        return charts
    
    def generate_facet_grids_its_by_wlb(self, save_path=None):
        """Plot 4: FacetGrid — Distribution of ITS1-ITS5 by WLB1 to WLB3"""
        charts = []
        
        self.df.columns = [c.strip() for c in self.df.columns]
        
        its_list = [f"ITS{i}" for i in range(1, 6)]
        wlb_list = [f"WLB{i}" for i in range(1, 4)]
        likert_order = ['1', '2', '3', '4', '5']
        
        for its in its_list:
            df_wlb = self.df.melt(
                id_vars=its,
                value_vars=wlb_list,
                var_name='WLB_Question',
                value_name='Response'
            )
            
            df_wlb[its] = pd.Categorical(df_wlb[its].astype(str), categories=likert_order, ordered=True)
            df_wlb['Response'] = pd.Categorical(df_wlb['Response'].astype(str), categories=likert_order, ordered=True)
            
            g = sns.catplot(
                data=df_wlb, kind='count',
                x='Response', hue=its,
                col='WLB_Question', col_wrap=2,
                height=4, aspect=1.5,
                palette='Set2',
                order=likert_order
            )
            
            g.fig.subplots_adjust(top=0.9)
            g.fig.suptitle(f'Distribution of {its} by WLB1–WLB3', fontsize=16)
            g.set_axis_labels('Response (Likert Scale)', 'Count')
            g.set_titles('{col_name}')
            g.add_legend(title=its)
            
            if save_path:
                filename = f"{save_path}/facetgrid_{its}_by_WLB.png"
                g.savefig(filename, dpi=150, bbox_inches='tight')
                charts.append(filename)
            plt.close()
        
        return charts
    
    def run_basic_eda(self, save_plots=False, output_path=None):
        """Run complete basic EDA pipeline"""
        print("=" * 60)
        print("STARTING BASIC EDA")
        print("=" * 60)
        
        # Load and inspect
        self.load_data()
        self.inspect_dataframe()
        self.separate_columns()
        
        # Handle missing values
        self.check_missing_values()
        self.impute_missing_values()
        
        # Generate visualizations
        if save_plots and output_path:
            print("\n Generating bar charts...")
            self.generate_bar_charts(output_path)
            
            print("\nGenerating stacked bar charts...")
            self.generate_stacked_bar_charts(output_path)
        
        # Generate cross tables and pivot tables
        print("\nGenerating cross tables...")
        cross_tables = self.generate_cross_tables()
        
        print("\nDetecting variable blocks...")
        var_blocks = self.detect_variable_blocks()
        
        print("\nGenerating pivot tables...")
        pivot_tables = self.generate_pivot_tables()
        
        print("\n" + "=" * 60)
        print("BASIC EDA COMPLETE")
        print("=" * 60)
        
        return {
            'cross_tables': cross_tables,
            'pivot_tables': pivot_tables,
            'variable_blocks': var_blocks
        }
    
    def run_advanced_plots(self, save_plots=True, output_path=None):
        """Run advanced plotting pipeline"""
        print("=" * 60)
        print("GENERATING ADVANCED PLOTS")
        print("=" * 60)
        
        if save_plots and output_path:
            print("\nGenerating FacetGrids: ITS by Q...")
            self.generate_facet_grids_its_by_q(output_path)
            
            print("\nGenerating FacetGrids: ITS by POS...")
            self.generate_facet_grids_its_by_pos(output_path)
            
            print("\nGenerating FacetGrids: ITS by JS...")
            self.generate_facet_grids_its_by_js(output_path)
            
            print("\nGenerating FacetGrids: ITS by WLB...")
            self.generate_facet_grids_its_by_wlb(output_path)
        
        print("\n" + "=" * 60)
        print("ADVANCED PLOTS COMPLETE")
        print("=" * 60)
