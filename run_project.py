"""Run all project stages in sequence. Use after installing requirements and running generate_data.py."""
from src.data_pipeline import audit,clean_data,validate
from src.customer_models import rfm_and_segments,build_lapse_model
from src.report import sql_report,eda_report,part_c_report,part_d_report
def main():
    print("\n=== PART A: AUDIT / CLEAN / VALIDATE / SQL / EDA ===")
    audit(); clean_data(); validate(); sql_report(); eda_report()
    print("\n=== PART B: SEGMENTS AND LAPSE MODEL ===")
    rfm,diagnostics,profile,pca=rfm_and_segments()
    print("RFM:",rfm.shape,rfm.recency.median(),rfm.recency.min(),rfm.recency.max(),round(rfm.frequency.mean(),2),rfm.frequency.max(),round(rfm.monetary.mean(),2),round(rfm.monetary.max(),2))
    print("K-means diagnostics:\n",diagnostics.round(3).to_string(index=False))
    print("Segment profile:\n",profile.round(1).to_string(index=False)); print("PCA explained variance:",pca,pca.sum())
    print("Lapse model summary:",build_lapse_model())
    print("\n=== PART C: TRIAGE ==="); part_c_report()
    print("\n=== PART D: POLICY RAG AND REFUND AGENT ==="); part_d_report()
    print("\nProject pipeline finished.")
if __name__=="__main__": main()
