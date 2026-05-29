# nodis/enrich/databases.py
"""
Database registry mapping biological level -> enrichment database names
per backend.

Biological levels
-----------------
rna                mRNA/gene-level: GO terms, KEGG, Reactome
post_transcriptional   tRNA/miRNA/RBP: miRNA targets, TF motifs, RNA-binding
protein            Protein complexes (CORUM), domains (InterPro), HPA
all                Union of all three levels
"""
from __future__ import annotations

VALID_LEVELS = ("rna", "post_transcriptional", "protein", "all")

# g:Profiler source identifiers
_GPROFILER: dict[str, list[str]] = {
    "rna": [
        "GO:BP",   # Gene Ontology - Biological Process
        "GO:MF",   # Gene Ontology - Molecular Function
        "GO:CC",   # Gene Ontology - Cellular Component
        "KEGG",    # KEGG pathways
        "REAC",    # Reactome
        "WP",      # WikiPathways
    ],
    "post_transcriptional": [
        "MIRNA",      # miRNA targets (miRTarBase)
        "TF",         # Transcription factor binding motifs (TRANSFAC/JASPAR)
        "MSIGDB:C3",  # miRNA target gene sets (MSigDB C3 collection)
    ],
    "protein": [
        "CORUM",      # Protein complexes
        "HP",         # Human Phenotype Ontology
        "HPA",        # Human Protein Atlas - tissue expression
        "MSIGDB:C4",  # Cancer gene neighborhoods
    ],
}

# GSEApy / Enrichr library names
_GSEAPY: dict[str, list[str]] = {
    "rna": [
        "GO_Biological_Process_2023",
        "GO_Molecular_Function_2023",
        "GO_Cellular_Component_2023",
        "KEGG_2021_Human",
        "Reactome_2022",
        "WikiPathways_2023_Human",
        "MSigDB_Hallmark_2020",
    ],
    "post_transcriptional": [
        "miRTarBase_2017",
        "TargetScan_microRNA_2017",
        "ENCODE_and_ChEA_Consensus_TFs_from_ChIP-X",
        "RNA-Seq_Disease_Gene_and_Drug_Signatures_from_GEO",
    ],
    "protein": [
        "CORUM",
        "PPI_Hub_Proteins",
        "InterPro_Domains_2019",
        "Human_Proteome_Map_Gene_Expression_Profiles_in_Normal_Tissues",
        "DIP_Kinase_Substrates_from_PhosphoSitePlus",
    ],
}

_REGISTRY: dict[str, dict[str, list[str]]] = {
    "gprofiler": _GPROFILER,
    "gseapy": _GSEAPY,
}


def get_databases(level: str, backend: str) -> list[str]:
    """Return the list of database identifiers for a given level and backend.

    Parameters
    ----------
    level : str
        One of ``"rna"``, ``"post_transcriptional"``, ``"protein"``, ``"all"``.
    backend : str
        One of ``"gprofiler"`` or ``"gseapy"``.

    Returns
    -------
    list of str
        Database / source identifiers for the selected level and backend.

    Raises
    ------
    ValueError
        If ``level`` or ``backend`` is not recognized.
    """
    if level not in VALID_LEVELS:
        raise ValueError(
            f"level must be one of {VALID_LEVELS}; got '{level}'."
        )
    if backend not in _REGISTRY:
        raise ValueError(
            f"backend must be 'gprofiler' or 'gseapy'; got '{backend}'."
        )

    db_map = _REGISTRY[backend]
    if level == "all":
        seen: list[str] = []
        for k in ("rna", "post_transcriptional", "protein"):
            for db in db_map[k]:
                if db not in seen:
                    seen.append(db)
        return seen

    return list(db_map[level])
