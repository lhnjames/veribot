import os
import sys
import json

def format_cell(mean_val, std_val=None, is_pct=True, bold=False):
    if mean_val is None:
        return "--"
    multiplier = 100.0 if is_pct else 1.0
    m_str = f"{mean_val * multiplier:.2f}"
    if std_val is not None and std_val > 0.0:
        s_str = f" \\pm {std_val * multiplier:.2f}"
    else:
        s_str = ""
    res = f"${m_str}{s_str}$"
    if bold:
        res = f"\\mathbf{{{res}}}"
    return res

def generate_latex_table(mgtab_summary, twibot_summary, output_path):
    variants = [
        ("A0_Full_VeriBot", "Full VeriBot (Bidirectional)", "Dim A: Architecture"),
        ("A1_Homogeneous_Challenger", "Homogeneous Challenger ($M_{\\text{fwd}} \\parallel M_{\\text{fwd}}$)", "Dim A: Architecture"),
        ("A2_Homogeneous_Ensemble", "Homogeneous Ensemble (Averaging)", "Dim A: Architecture"),
        ("A3_Unidirectional_Self_Search", "Unidirectional Self-Search (Single-Model)", "Dim A: Architecture"),
        ("B1_Residual_Only", "Residual Graph Only ($V \\setminus C^*$)", "Dim B: Search Space"),
        ("B2_FullGraph_CrossAttention", "Full-Graph Cross-Attention", "Dim B: Search Space"),
        ("B3_Matched_Random_Control", "Matched Random Control", "Dim B: Search Space"),
        ("C1_Unary_Only", "Unary-Only (No Synergy Pairs)", "Dim C: Representations"),
        ("C2_Leave_Out_Profile", "w/o Profile Evidence", "Dim C: Representations"),
        ("C3_Leave_Out_Text", "w/o Text Evidence", "Dim C: Representations"),
        ("C4_Leave_Out_Topology", "w/o Topology Evidence", "Dim C: Representations"),
        ("C5_Leave_Out_Temporal", "w/o Temporal Evidence", "Dim C: Representations"),
        ("D1_Dense_Unconstrained_Budget", "Dense Unconstrained Budget", "Dim D: Invariance & Budget"),
        ("D2_Standard_Biased_Projection", "Standard Biased Projection ($\\Phi(0) \\ne 0$)", "Dim D: Invariance & Budget"),
    ]
    
    lines = []
    lines.append("% -------------------------------------------------------------")
    lines.append("% VeriBot Comprehensive 14-Variant Ablation Table (5-Seed Official)")
    lines.append("% -------------------------------------------------------------")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\caption{\\textbf{Comprehensive 14-Variant Ablation Study across 4 Dimensions on MGTAB and TwiBot-20 (5-Seed Mean $\\pm$ Std).} "
                 "All models are calibrated strictly on Dev under Zero-Leakage protocols and evaluated on locked Test sets. "
                 "Macro-F1 is the primary official classification metric.}")
    lines.append("\\label{tab:ablation_master}")
    lines.append("\\resizebox{\\textwidth}{!}{")
    lines.append("\\begin{tabular}{l | cccc | cccc}")
    lines.append("\\toprule")
    lines.append("& \\multicolumn{4}{c|}{\\textbf{MGTAB Benchmark}} & \\multicolumn{4}{c}{\\textbf{TwiBot-20 Benchmark}} \\\\")
    lines.append("\\textbf{Ablation Variant} & \\textbf{Acc (\\%)} & \\textbf{Macro-F1 (\\%)} & \\textbf{Prec (\\%)} & \\textbf{Rec (\\%)} & \\textbf{Acc (\\%)} & \\textbf{Macro-F1 (\\%)} & \\textbf{Prec (\\%)} & \\textbf{Rec (\\%)} \\\\")
    lines.append("\\midrule")
    
    current_group = None
    for var_id, var_name, group in variants:
        if group != current_group:
            current_group = group
            lines.append(f"\\midrule")
            lines.append(f"\\multicolumn{{9}}{{l}}{\\textit{{\\textbf{{{group}}}}} \\\\")
            lines.append(f"\\midrule")
            
        is_baseline = (var_id == "A0_Full_VeriBot")
        
        # MGTAB data
        m_data = mgtab_summary.get(var_id, {})
        m_bi = m_data.get("bidirectional_refutation", {})
        m_acc = format_cell(m_bi.get("acc", {}).get("mean"), m_bi.get("acc", {}).get("std"), bold=is_baseline)
        m_f1 = format_cell(m_bi.get("macro_f1", {}).get("mean"), m_bi.get("macro_f1", {}).get("std"), bold=is_baseline)
        m_pre = format_cell(m_bi.get("precision", {}).get("mean"), m_bi.get("precision", {}).get("std"), bold=is_baseline)
        m_rec = format_cell(m_bi.get("recall", {}).get("mean"), m_bi.get("recall", {}).get("std"), bold=is_baseline)
        
        # TwiBot data
        t_data = twibot_summary.get(var_id, {})
        t_bi = t_data.get("bidirectional_refutation", {})
        t_acc = format_cell(t_bi.get("acc", {}).get("mean"), t_bi.get("acc", {}).get("std"), bold=is_baseline)
        t_f1 = format_cell(t_bi.get("macro_f1", {}).get("mean"), t_bi.get("macro_f1", {}).get("std"), bold=is_baseline)
        t_pre = format_cell(t_bi.get("precision", {}).get("mean"), t_bi.get("precision", {}).get("std"), bold=is_baseline)
        t_rec = format_cell(t_bi.get("recall", {}).get("mean"), t_bi.get("recall", {}).get("std"), bold=is_baseline)
        
        row = f"{var_name} & {m_acc} & {m_f1} & {m_pre} & {m_rec} & {t_acc} & {t_f1} & {t_pre} & {t_rec} \\\\"
        lines.append(row)
        
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table*}")
    
    content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Generated LaTeX ablation table at {output_path}")

if __name__ == "__main__":
    mgtab_p = sys.argv[1] if len(sys.argv) > 1 else "runs/ablation/mgtab/ablation_master_summary.json"
    twibot_p = sys.argv[2] if len(sys.argv) > 2 else "runs/ablation/twibot20/ablation_master_summary.json"
    out_p = sys.argv[3] if len(sys.argv) > 3 else "runs/ablation/ablation_table.tex"
    
    m_dict = {}
    t_dict = {}
    if os.path.exists(mgtab_p):
        with open(mgtab_p, "r", encoding="utf-8") as f:
            m_dict = json.load(f)
    if os.path.exists(twibot_p):
        with open(twibot_p, "r", encoding="utf-8") as f:
            t_dict = json.load(f)
            
    generate_latex_table(m_dict, t_dict, out_p)
