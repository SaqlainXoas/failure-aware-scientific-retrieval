# Structured Calibration-Dev Failure Analysis

These 12 hybrid-RRF cases were selected by the predefined confidence-ranking rules in `results/tables/failure_case_selection.json`, before reading case content. Scores and ranks below are rendered from saved run artifacts; interpretations are retrieval-error descriptions, not support/refute judgments.

- Split: `calibration-dev`
- Pipeline: `hybrid_rrf`
- Confidence model: `calibrated` fitted on `calibration-train`
- Confidence source: `2026-08-05T114843.983136Z_hybrid_rrf_calibration-train`
- Ranking source: `2026-08-05T114749.452090Z_hybrid_rrf_calibration-dev`

## 1. Query `398` — high_confidence_incorrect

**Claim:** Exhaustion of B cells contributes to poor Ab response in HIV-infected individuals.

- Transition: `unchanged_failure`
- Selection reason: incorrect confidence rank 1 of 20
- Common-feature confidence: 0.823306

### Representative gold evidence

**Antibody-Based HIV-1 Vaccines: Recent Developments and Future Directions** (`8883846`)

> immunoregulation of B cell responses.

Saved position: first stage 34; reranked 32.

### First-stage top five

| Rank | Document | Score | Gold |
| ---: | --- | ---: | :---: |
| 1 | Decoupling activation and exhaustion of B cells in spontaneous controllers of HIV infection. (`20261352`) | 0.031099 |  |
| 2 | Apoptosis occurs predominantly in bystander cells and not in productively infected cells of HIV- and SIV-infected lymph nodes (`37444589`) | 0.030090 |  |
| 3 | Immune activation and HIV persistence: implications for curative approaches to HIV infection. (`44562058`) | 0.030018 |  |
| 4 | HIV–1 Infects Multipotent Progenitor Cells Causing Cell Death and Establishing Latent Cellular Reservoirs (`7224723`) | 0.029851 |  |
| 5 | Characterization of Programmed Death-1 Homologue-1 (PD-1H) Expression and Function in Normal and HIV Infected Individuals (`5752492`) | 0.029551 |  |

### Reranked top five

| Rank | Document | Score | Gold |
| ---: | --- | ---: | :---: |
| 1 | Decoupling activation and exhaustion of B cells in spontaneous controllers of HIV infection. (`20261352`) | 3.974309 |  |
| 2 | CD8 Cells of Patients with Diffuse Cutaneous Leishmaniasis Display Functional Exhaustion: The Latter Is Reversed, In Vitro, by TLR2 Agonists (`14657344`) | -0.720793 |  |
| 3 | Simian immunodeficiency virus–induced mucosal interleukin-17 deficiency promotes Salmonella dissemination from the gut (`29023309`) | -1.234845 |  |
| 4 | CCL2/monocyte chemoattractant protein-1 mediates enhanced transmigration of human immunodeficiency virus (HIV)-infected leukocytes across the blood-brain barrier: a potential mechanism of HIV-CNS invasion and NeuroAIDS. (`27602752`) | -1.334070 |  |
| 5 | Immunophenotypic analysis of AIDS-related diffuse large B-cell lymphoma and clinical implications in patients from AIDS malignancies consortium clinical trials 010 and 034 (`23304931`) | -1.657244 |  |

### Interpretation

Labels: `title bias`, `cross-encoder overconfidence`.

The highest-ranked non-gold abstract names B-cell exhaustion and HIV directly, while the annotated review discusses antibody responses more broadly. The reranker strongly favors the near-verbatim title match and leaves the gold review well below the cutoff.

## 2. Query `4` — high_confidence_incorrect

**Claim:** 1-1% of colorectal cancer patients are diagnosed with regional or distant metastases.

- Transition: `degraded_by_reranker`
- Selection reason: incorrect confidence rank 2 of 20
- Common-feature confidence: 0.622160

### Representative gold evidence

**Relation between Medicare screening reimbursement and stage at diagnosis for older patients with colon cancer.** (`22942787`)

> The proportion of patients diagnosed at an early stage increased from 22.5% in period 1 to 25.5% in period 2 and 26.3% in period 3

Saved position: first stage 5; reranked 16.

### First-stage top five

| Rank | Document | Score | Gold |
| ---: | --- | ---: | :---: |
| 1 | Nationwide trends in incidence, treatment and survival of colorectal cancer patients with synchronous metastases (`10958594`) | 0.032787 |  |
| 2 | Social and geographical factors affecting access to treatment of colorectal cancer: a cancer registry study (`5641851`) | 0.031754 |  |
| 3 | Cancer survival increases in Europe, but international differences remain wide. (`30226988`) | 0.030769 |  |
| 4 | Colorectal cancer survival in socioeconomic groups in England: variation is mainly in the short term after diagnosis. (`2058909`) | 0.030159 |  |
| 5 | Relation between Medicare screening reimbursement and stage at diagnosis for older patients with colon cancer. (`22942787`) | 0.029324 | yes |

### Reranked top five

| Rank | Document | Score | Gold |
| ---: | --- | ---: | :---: |
| 1 | Nationwide trends in incidence, treatment and survival of colorectal cancer patients with synchronous metastases (`10958594`) | 3.974902 |  |
| 2 | Surgical Resection for Patients with Solid Brain Metastases: Current Status (`23639838`) | 1.326813 |  |
| 3 | Malignancies, prothrombotic mutations, and the risk of venous thrombosis. (`1387104`) | 1.248182 |  |
| 4 | Operable non-small cell lung cancer diagnosed by transpleural techniques : do they affect relapse and prognosis? (`18062308`) | 0.677462 |  |
| 5 | Hypoxia in relation to vasculature and proliferation in liver metastases in patients with colorectal cancer. (`24980622`) | 0.456289 |  |

### Interpretation

Labels: `entity/numerical mismatch`, `cross-encoder overconfidence`.

The claim is dominated by a malformed percentage and metastatic-stage wording. A metastasis-focused non-gold abstract receives the top reranker score, while the annotated stage-at-diagnosis study falls from the first-stage top five to below the final cutoff.

## 3. Query `610` — candidate_set_failure

**Claim:** Increased flux of microbial products suppresses immune responses.

- Transition: `no_opportunity`
- Selection reason: confidence-spread index 0 of 5 eligible no_opportunity cases
- Common-feature confidence: 0.116313

### Representative gold evidence

**Compromised intestinal epithelial barrier induces adaptive immune compensation that protects from colitis.** (`40096222`)

> These data establish a role for adaptive immune-mediated protection from acute colitis under conditions of intestinal epithelial barrier compromise.

Saved position: first stage 83; reranked not in top-50.

### First-stage top five

| Rank | Document | Score | Gold |
| ---: | --- | ---: | :---: |
| 1 | Beyond pattern recognition: five immune checkpoints for scaling the microbial threat (`21439293`) | 0.031054 |  |
| 2 | Foxp3(+) T cells regulate immunoglobulin a selection and facilitate diversification of bacterial species responsible for immune homeostasis. (`42693833`) | 0.030159 |  |
| 3 | Microbiome-driven allergic lung inflammation is ameliorated by short-chain fatty acids (`20148808`) | 0.027347 |  |
| 4 | Programmed death-1–induced interleukin-10 production by monocytes impairs CD4+ T cell activation during HIV infection (`10648422`) | 0.026993 |  |
| 5 | Intestinal bacteria and the regulation of immune cell homeostasis. (`26702468`) | 0.026655 |  |

### Reranked top five

| Rank | Document | Score | Gold |
| ---: | --- | ---: | :---: |
| 1 | TLR-activated B cells suppress T cell-mediated autoimmunity. (`14644164`) | 0.674132 |  |
| 2 | The inflammasome component NLRP3 impairs antitumor vaccine by enhancing the accumulation of tumor-associated myeloid-derived suppressor cells. (`15435343`) | -1.768179 |  |
| 3 | Regulation of the antimicrobial response by NLR proteins. (`4664540`) | -2.116372 |  |
| 4 | Candida albicans-Staphylococcus aureus polymicrobial peritonitis modulates host innate immunity. (`45401535`) | -2.421447 |  |
| 5 | Regulation of innate immunity by NADPH oxidase. (`21535641`) | -2.606775 |  |

### Interpretation

Labels: `terminology mismatch`, `evidence outside the cutoff`.

The claim compresses barrier leakage, microbial translocation, and compensatory immune protection into the phrase “microbial products.” Broader immune-suppression papers dominate retrieval, while the annotated barrier-compromise study appears only beyond the top-50 candidate set.

## 4. Query `1096` — candidate_set_failure

**Claim:** Specialized functional cell types can be derived from human pluripotent stem cells.

- Transition: `no_opportunity`
- Selection reason: confidence-spread index 2 of 5 eligible no_opportunity cases
- Common-feature confidence: 0.517393

### Representative gold evidence

**Complex Tissue and Disease Modeling using hiPSCs.** (`29638116`)

> Defined genetic models based on human pluripotent stem cells have opened new avenues for understanding disease mechanisms and drug screening.

Saved position: first stage 60; reranked not in top-50.

### First-stage top five

| Rank | Document | Score | Gold |
| ---: | --- | ---: | :---: |
| 1 | Small molecule screening in human induced pluripotent stem cell-derived terminal cell types. (`28386343`) | 0.031319 |  |
| 2 | Induction of human neuronal cells by defined transcription factors (`4405194`) | 0.030579 |  |
| 3 | Induced pluripotent stem cell lines derived from human somatic cells. (`86129154`) | 0.030118 |  |
| 4 | Generation of induced pluripotent stem cells from Asian patients with chronic neurodegenerative diseases. (`35777860`) | 0.029040 |  |
| 5 | Direct conversion of human fibroblasts to multilineage blood progenitors (`4417177`) | 0.028778 |  |

### Reranked top five

| Rank | Document | Score | Gold |
| ---: | --- | ---: | :---: |
| 1 | Epigenetic memory and preferential lineage-specific differentiation in induced pluripotent stem cells derived from human pancreatic islet beta cells. (`27588420`) | 6.845609 |  |
| 2 | Small molecule screening in human induced pluripotent stem cell-derived terminal cell types. (`28386343`) | 6.631100 |  |
| 3 | Derivation of novel human ground state naive pluripotent stem cells (`4462419`) | 6.284543 |  |
| 4 | Induced pluripotent stem cell lines derived from human somatic cells. (`86129154`) | 6.018391 |  |
| 5 | Human oocytes reprogram adult somatic nuclei of a type 1 diabetic to diploid pluripotent stem cells (`4457834`) | 5.889598 |  |

### Interpretation

Labels: `partial topical relevance`, `evidence outside the cutoff`.

Many corpus abstracts describe individual pluripotent-cell derivations and therefore match the broad claim closely. The annotated paper instead emphasizes complex multicellular tissue models, so its less literal framing is pushed outside the reranker’s candidate set.

## 5. Query `92` — candidate_set_failure

**Claim:** Aged patients are more susceptible to ischaemia/reperfusion injury.

- Transition: `no_opportunity`
- Selection reason: confidence-spread index 4 of 5 eligible no_opportunity cases
- Common-feature confidence: 0.531400

### Representative gold evidence

**Restoration of chaperone-mediated autophagy in aging liver improves cellular maintenance and hepatic function** (`1084345`)

> CMA activity declines in aged organisms and have proposed that this failure in cellular clearance could contribute to the accumulation of altered proteins

Saved position: first stage not in top-100; reranked not in top-50.

### First-stage top five

| Rank | Document | Score | Gold |
| ---: | --- | ---: | :---: |
| 1 | Molecular mechanisms of hepatic ischemia-reperfusion injury and preconditioning. (`6580081`) | 0.032258 |  |
| 2 | Comparative Safety of Targeted Therapies for Metastatic Colorectal Cancer between Elderly and Younger Patients: a Study Using the International Pharmacovigilance Database (`2492146`) | 0.027826 |  |
| 3 | Sphingosine can pre- and post-condition heart and utilizes a different mechanism from sphingosine 1-phosphate. (`43700577`) | 0.024206 |  |
| 4 | Social variations in access to hospital care for patients with colorectal, breast, and lung cancer between 1999 and 2006: retrospective analysis of hospital episode statistics (`16390264`) | 0.023699 |  |
| 5 | Impact of blood pressure variability on cardiovascular events in elderly patients with hypertension. (`54490092`) | 0.023546 |  |

### Reranked top five

| Rank | Document | Score | Gold |
| ---: | --- | ---: | :---: |
| 1 | Plasma homocysteine is a risk factor for recurrent vascular events in young patients with an ischaemic stroke or TIA (`9813098`) | 0.390088 |  |
| 2 | Molecular mechanisms of hepatic ischemia-reperfusion injury and preconditioning. (`6580081`) | -1.789206 |  |
| 3 | Comparative Safety of Targeted Therapies for Metastatic Colorectal Cancer between Elderly and Younger Patients: a Study Using the International Pharmacovigilance Database (`2492146`) | -2.474385 |  |
| 4 | Impact of blood pressure variability on cardiovascular events in elderly patients with hypertension. (`54490092`) | -3.818043 |  |
| 5 | The role of free radicals in cold injuries. (`13000926`) | -4.075304 |  |

### Interpretation

Labels: `lexical distraction`, `partial topical relevance`.

The query combines aging with ischemia/reperfusion, causing direct ischemia titles to dominate. The annotated aging-liver study describes declining cellular maintenance rather than ischemic injury explicitly, and it is absent even from the saved top-100.

## 6. Query `572` — reranking_failure

**Claim:** In chronic viral infections or tumors, peptides that selectively inhibit PTPRS can be utilized to boost insufficient activity of pDCs.

- Transition: `unchanged_failure`
- Selection reason: confidence-spread index 1 of 8 eligible unchanged_failure cases
- Common-feature confidence: 0.140124

### Representative gold evidence

**Modulation of the proteoglycan receptor PTPσ promotes recovery after spinal cord injury** (`4447055`)

> We generated a membrane-permeable peptide mimetic of the PTPσ wedge domain that binds to PTPσ and relieves CSPG-mediated inhibition.

Saved position: first stage 17; reranked 16.

### First-stage top five

| Rank | Document | Score | Gold |
| ---: | --- | ---: | :---: |
| 1 | In situ regulation of DC subsets and T cells mediates tumor regression in mice. (`13231899`) | 0.030835 |  |
| 2 | Defective tryptophan catabolism underlies inflammation in mouse chronic granulomatous disease (`4391121`) | 0.025149 |  |
| 3 | Programmed death-1–induced interleukin-10 production by monocytes impairs CD4+ T cell activation during HIV infection (`10648422`) | 0.025026 |  |
| 4 | Comparison of the intrinsic kinase activity and substrate specificity of c-Abl and Bcr-Abl. (`30379039`) | 0.024383 |  |
| 5 | A specific inhibitor of phosphatidylinositol 3-kinase, 2-(4-morpholinyl)-8-phenyl-4H-1-benzopyran-4-one (LY294002). (`19752008`) | 0.022964 |  |

### Reranked top five

| Rank | Document | Score | Gold |
| ---: | --- | ---: | :---: |
| 1 | Anti-Angiogenic Activity of Selected Receptor Tyrosine Kinase Inhibitors, PD166285 and PD173074: Implications for Combination Treatment with Photodynamic Therapy (`23420807`) | -1.017085 |  |
| 2 | In situ regulation of DC subsets and T cells mediates tumor regression in mice. (`13231899`) | -2.065557 |  |
| 3 | Inhibition of SRC expression and activity inhibits tumor progression and metastasis of human pancreatic adenocarcinoma cells in an orthotopic nude mouse model. (`45764440`) | -2.191133 |  |
| 4 | Programmed death-1–induced interleukin-10 production by monocytes impairs CD4+ T cell activation during HIV infection (`10648422`) | -2.203836 |  |
| 5 | CD4+ T cell-mediated cytotoxicity eliminates primary tumor cells in metastatic melanoma through high MHC class II expression and can be enhanced by inhibitory receptor blockade (`15128866`) | -2.346689 |  |

### Interpretation

Labels: `entity/numerical mismatch`, `evidence outside the cutoff`.

The PTPRS/PTPσ identity connects the claim to the annotated peptide study, but the surrounding immune and tumor terminology points toward unrelated inhibitor papers. Reranking improves the gold position only slightly and leaves it outside the top ten.

## 7. Query `422` — reranking_failure

**Claim:** Flexible molecules experience less steric hindrance in the tumor microenviroment than rigid molecules.

- Transition: `unchanged_failure`
- Selection reason: confidence-spread index 4 of 8 eligible unchanged_failure cases
- Common-feature confidence: 0.216174

### Representative gold evidence

**Quantum dots spectrally distinguish multiple species within the tumor milieu in vivo** (`11172205`)

> we used them to measure the ability of particles of different sizes to access the tumor.

Saved position: first stage 33; reranked 20.

### First-stage top five

| Rank | Document | Score | Gold |
| ---: | --- | ---: | :---: |
| 1 | Use of TLS parameters to model anisotropic displacements in macromolecular refinement. (`39728826`) | 0.024658 |  |
| 2 | Chemical approaches to stem cell biology and therapeutics. (`502797`) | 0.024548 |  |
| 3 | Extracellular vesicles derived from renal cancer stem cells induce a pro-tumorigenic phenotype in mesenchymal stromal cells (`14768471`) | 0.021719 |  |
| 4 | Metabolic phenotyping of human blood plasma: a powerful tool to discriminate between cancer types? (`17163294`) | 0.017168 |  |
| 5 | Dynamics of Microvillus Extension and Tether Formation in Rolling Leukocytes. (`43224840`) | 0.017095 |  |

### Reranked top five

| Rank | Document | Score | Gold |
| ---: | --- | ---: | :---: |
| 1 | Microenvironment rigidity modulates responses to the HER2 receptor tyrosine kinase inhibitor lapatinib via YAP and TAZ transcription factors. (`1065627`) | -1.877039 |  |
| 2 | Microenvironmental regulation of tumor progression and metastasis (`6944800`) | -5.066760 |  |
| 3 | Cancer-related inflammation (`4429118`) | -5.804200 |  |
| 4 | Mechanical regulation of cell function with geometrically modulated elastomeric substrates (`17388232`) | -6.450350 |  |
| 5 | Combination cancer immunotherapies tailored to the tumour microenvironment (`24726600`) | -7.129634 |  |

### Interpretation

Labels: `terminology mismatch`, `evidence outside the cutoff`.

The annotated experiment operationalizes steric access through particle size, whereas the claim uses flexibility and rigidity language. The reranker promotes abstracts that literally mention microenvironment rigidity, improving the gold rank but not enough to cross the cutoff.

## 8. Query `1212` — reranking_failure

**Claim:** The deregulated and prolonged activation of monocytes has deleterious effects in chronic infectious conditions.

- Transition: `unchanged_failure`
- Selection reason: confidence-spread index 6 of 8 eligible unchanged_failure cases
- Common-feature confidence: 0.396642

### Representative gold evidence

**Kruppel-like factor 2 is a transcriptional regulator of chronic and acute inflammation.** (`44724517`)

> Although myeloid cell activation is requisite for an optimal innate immune response, this process must be tightly controlled to prevent collateral host tissue damage.

Saved position: first stage 45; reranked 28.

### First-stage top five

| Rank | Document | Score | Gold |
| ---: | --- | ---: | :---: |
| 1 | TLR activation triggers the rapid differentiation of monocytes into macrophages and dendritic cells (`21498497`) | 0.032522 |  |
| 2 | CCR2 and CXCR4 regulate peripheral blood monocyte pharmacodynamics and link to efficacy in experimental autoimmune encephalomyelitis (`17934603`) | 0.031545 |  |
| 3 | Programmed death-1–induced interleukin-10 production by monocytes impairs CD4+ T cell activation during HIV infection (`10648422`) | 0.030536 |  |
| 4 | Characterization of Programmed Death-1 Homologue-1 (PD-1H) Expression and Function in Normal and HIV Infected Individuals (`5752492`) | 0.030331 |  |
| 5 | Subpopulations of mouse blood monocytes differ in maturation stage and inflammatory response. (`36444198`) | 0.030118 |  |

### Reranked top five

| Rank | Document | Score | Gold |
| ---: | --- | ---: | :---: |
| 1 | Programmed death-1–induced interleukin-10 production by monocytes impairs CD4+ T cell activation during HIV infection (`10648422`) | 0.617253 |  |
| 2 | Characterization of Programmed Death-1 Homologue-1 (PD-1H) Expression and Function in Normal and HIV Infected Individuals (`5752492`) | 0.231836 |  |
| 3 | TLR activation triggers the rapid differentiation of monocytes into macrophages and dendritic cells (`21498497`) | 0.184016 |  |
| 4 | Toll-like receptor-3 mediates HIV-1 transactivation via NFκB and JNK pathways and histone acetylation, but prolonged activation suppresses Tat and HIV-1 replication. (`25606339`) | -1.710496 |  |
| 5 | Ly6Chi monocytes direct alternatively activated profibrotic macrophage regulation of lung fibrosis. (`22621251`) | -1.803354 |  |

### Interpretation

Labels: `lexical distraction`, `evidence outside the cutoff`.

Monocyte, chronic infection, and prolonged activation terms steer both stages toward HIV-specific abstracts. The annotated KLF2 study describes the more general mechanism of harmful, insufficiently controlled myeloid inflammation and remains far below the top ten.

## 9. Query `1325` — reranker_rescue

**Claim:** Treatment with the EC uptake inhibitor AM404 resulted in a dose-dependent decrease in the expression of immobility.

- Transition: `rescued_by_reranker`
- Selection reason: confidence-spread index 1 of 7 eligible rescued_by_reranker cases
- Common-feature confidence: 0.152612

### Representative gold evidence

**Functional role of high-affinity anandamide transport, as revealed by selective inhibition.** (`40476126`)

> The compound N-(4-hydroxyphenyl)arachidonylamide (AM404) was shown to inhibit high-affinity anandamide accumulation in rat neurons and astrocytes in vitro

Saved position: first stage 16; reranked 7.

### First-stage top five

| Rank | Document | Score | Gold |
| ---: | --- | ---: | :---: |
| 1 | A phase I study of PF-04449913, an oral hedgehog inhibitor, in patients with advanced solid tumors. (`14626540`) | 0.028191 |  |
| 2 | Therapeutic potential of GSK-J4, a histone demethylase KDM6B/JMJD3 inhibitor, for acute myeloid leukemia (`4387494`) | 0.027200 |  |
| 3 | Inhibition of fatty acid oxidation modulates immunosuppressive functions of myeloid-derived suppressor cells and enhances cancer therapies (`2030623`) | 0.024706 |  |
| 4 | Randomized dose-finding clinical trial of oncolytic immunotherapeutic vaccinia JX-594 in liver cancer (`27437459`) | 0.023985 |  |
| 5 | Inhibition of apical sodium-dependent bile acid transporter as a novel treatment for diabetes. (`12280462`) | 0.022684 |  |

### Reranked top five

| Rank | Document | Score | Gold |
| ---: | --- | ---: | :---: |
| 1 | Feedback upregulation of HER3 (ErbB3) expression and activity attenuates antitumor effect of PI3K inhibitors. (`23863551`) | 1.105883 |  |
| 2 | β-site amyloid precursor protein-cleaving enzyme 1(BACE1) inhibitor treatment induces Aβ5-X peptides through alternative amyloid precursor protein cleavage (`10207180`) | 0.475888 |  |
| 3 | Immunosuppressant FTY720 inhibits thymocyte emigration. (`1336292`) | -0.875118 |  |
| 4 | Therapeutic potential of GSK-J4, a histone demethylase KDM6B/JMJD3 inhibitor, for acute myeloid leukemia (`4387494`) | -0.928175 |  |
| 5 | Simvastatin reduces CD40 expression in an experimental model of early arterialization of saphenous vein graft. (`6212802`) | -1.404510 |  |

### Interpretation

Labels: `terminology mismatch`, `evidence outside the cutoff`.

The first stage is distracted by generic inhibitor language and does not place the AM404 study near the top. Pairwise reranking recognizes the exact compound and uptake mechanism strongly enough to move the gold document into the final top ten.

## 10. Query `955` — reranker_rescue

**Claim:** Pioneer factor OCT3/4 interacts with major chromatin remodeling factors.

- Transition: `rescued_by_reranker`
- Selection reason: confidence-spread index 5 of 7 eligible rescued_by_reranker cases
- Common-feature confidence: 0.428708

### Representative gold evidence

**Oct4 links multiple epigenetic pathways to the pluripotency network** (`2078658`)

> we found that Oct4 is associated with multiple chromatin-modifying complexes with documented as well as newly proved functional significance in stem cell maintenance

Saved position: first stage 29; reranked 6.

### First-stage top five

| Rank | Document | Score | Gold |
| ---: | --- | ---: | :---: |
| 1 | Pioneer Transcription Factors Target Partial DNA Motifs on Nucleosomes to Initiate Reprogramming (`11011905`) | 0.032522 |  |
| 2 | Pioneer factor Pax7 deploys a stable enhancer repertoire for specification of cell fate (`4896726`) | 0.031545 |  |
| 3 | Catalytic-Independent Functions of PARP-1 Determine Sox2 Pioneer Activity at Intractable Genomic Loci. (`23208167`) | 0.030798 |  |
| 4 | Epigenetic switch involved in activation of pioneer factor FOXA1-dependent enhancers. (`2151983`) | 0.030310 |  |
| 5 | Facilitators and Impediments of the Pluripotency Reprogramming Factors' Initial Engagement with the Genome (`18998807`) | 0.030118 |  |

### Reranked top five

| Rank | Document | Score | Gold |
| ---: | --- | ---: | :---: |
| 1 | Pioneer Transcription Factors Target Partial DNA Motifs on Nucleosomes to Initiate Reprogramming (`11011905`) | 4.123984 |  |
| 2 | Pioneer factor Pax7 deploys a stable enhancer repertoire for specification of cell fate (`4896726`) | 3.671358 |  |
| 3 | Wdr5 Mediates Self-Renewal and Reprogramming via the Embryonic Stem Cell Core Transcriptional Network (`14191255`) | 3.171580 |  |
| 4 | Facilitators and Impediments of the Pluripotency Reprogramming Factors' Initial Engagement with the Genome (`18998807`) | 2.553157 |  |
| 5 | An Oct4-Centered Protein Interaction Network in Embryonic Stem Cells (`30507607`) | 2.526716 | yes |

### Interpretation

Labels: `lexical distraction`, `evidence outside the cutoff`.

Pioneer-factor terminology initially favors general chromatin-engagement papers, leaving the representative gold abstract deep in the candidates. Reranking responds to the explicit Oct4/chromatin-complex relationship and rescues annotated evidence into the final top ten.

## 11. Query `655` — reranker_degradation

**Claim:** Intra-cerebroventricular infusion of amyloid-β oligomers reduces expression of fibronectin type-III domain-containing protein 5 mRNA in mice hippocampi.

- Transition: `degraded_by_reranker`
- Selection reason: confidence-spread index 1 of 5 eligible degraded_by_reranker cases
- Common-feature confidence: 0.070887

### Representative gold evidence

**Exercise-linked FNDC5/irisin rescues synaptic plasticity and memory defects in Alzheimer’s models** (`57574395`)

> Here we show that FNDC5/irisin levels are reduced in AD hippocampi and cerebrospinal fluid, and in experimental AD models.

Saved position: first stage 1; reranked 13.

### First-stage top five

| Rank | Document | Score | Gold |
| ---: | --- | ---: | :---: |
| 1 | Exercise-linked FNDC5/irisin rescues synaptic plasticity and memory defects in Alzheimer’s models (`57574395`) | 0.032787 | yes |
| 2 | A specific amyloid-β protein assembly in the brain impairs memory (`4407385`) | 0.030536 |  |
| 3 | Phagocytosis and deposition of vascular beta-amyloid in rat brains injected with Alzheimer beta-amyloid. (`24989194`) | 0.028958 |  |
| 4 | Effects of Hypoxia and Oxidative Stress on Expression of Neprilysin in Human Neuroblastoma Cells and Rat Cortical Neurones and Astrocytes (`9194077`) | 0.028382 |  |
| 5 | Alzheimer’s Disease Risk Gene CD33 Inhibits Microglial Uptake of Amyloid Beta (`7221410`) | 0.027501 |  |

### Reranked top five

| Rank | Document | Score | Gold |
| ---: | --- | ---: | :---: |
| 1 | INTRANASAL INSULIN IMPROVES COGNITION AND MODULATES β-AMYLOID IN EARLY AD (`10190462`) | -3.052739 |  |
| 2 | Targeting of cell-surface β-amyloid precursor protein to lysosomes: alternative processing into amyloid-bearing fragments (`4361990`) | -3.679794 |  |
| 3 | Effects of Hypoxia and Oxidative Stress on Expression of Neprilysin in Human Neuroblastoma Cells and Rat Cortical Neurones and Astrocytes (`9194077`) | -3.699770 |  |
| 4 | ADAM13 disintegrin and cysteine-rich domains bind to the second heparin-binding domain of fibronectin. (`20943272`) | -3.919012 |  |
| 5 | β-site amyloid precursor protein-cleaving enzyme 1(BACE1) inhibitor treatment induces Aβ5-X peptides through alternative amyloid precursor protein cleavage (`10207180`) | -4.085125 |  |

### Interpretation

Labels: `partial topical relevance`, `cross-encoder overconfidence`.

Hybrid retrieval correctly puts the FNDC5/irisin Alzheimer’s study first. Its abstract states the broader reduction in experimental models rather than the query’s precise infusion and mRNA wording, and the reranker instead promotes abstracts with more literal amyloid terminology.

## 12. Query `393` — reranker_degradation

**Claim:** Ethanol stress reduces the expression of SRL in bacteria.

- Transition: `degraded_by_reranker`
- Selection reason: confidence-spread index 3 of 5 eligible degraded_by_reranker cases
- Common-feature confidence: 0.336546

### Representative gold evidence

**Regulatory and metabolic rewiring during laboratory evolution of ethanol tolerance in E. coli** (`1148122`)

> We used fitness profiling to measure the consequences of single-locus perturbations in the context of ethanol exposure.

Saved position: first stage 2; reranked 14.

### First-stage top five

| Rank | Document | Score | Gold |
| ---: | --- | ---: | :---: |
| 1 | Complex physiology and compound stress responses during fermentation of alkali-pretreated corn stover hydrolysate by an Escherichia coli ethanologen. (`21602220`) | 0.032522 |  |
| 2 | Regulatory and metabolic rewiring during laboratory evolution of ethanol tolerance in E. coli (`1148122`) | 0.032266 | yes |
| 3 | Two-stage control of an oxidative stress regulon: the Escherichia coli SoxR protein triggers redox-inducible expression of the soxS regulatory gene. (`471735`) | 0.029236 |  |
| 4 | Stress and host immunity amplify Mycobacterium tuberculosis phenotypic heterogeneity and induce nongrowing metabolically active forms. (`22908536`) | 0.026496 |  |
| 5 | The first gene of the Bacillus subtilis clpC operon, ctsR, encodes a negative regulator of its own operon and other class III heat shock genes. (`34386619`) | 0.025129 |  |

### Reranked top five

| Rank | Document | Score | Gold |
| ---: | --- | ---: | :---: |
| 1 | Complex physiology and compound stress responses during fermentation of alkali-pretreated corn stover hydrolysate by an Escherichia coli ethanologen. (`21602220`) | -2.575015 |  |
| 2 | Effect of hyperosmolality on alkaline phosphatase and stress-response protein 27 of MCF-7 breast cancer cells (`1554348`) | -4.420338 |  |
| 3 | Two-stage control of an oxidative stress regulon: the Escherichia coli SoxR protein triggers redox-inducible expression of the soxS regulatory gene. (`471735`) | -4.682820 |  |
| 4 | Glutathione and glutathione-dependent enzymes represent a co-ordinately regulated defence against oxidative stress. (`25488034`) | -5.480850 |  |
| 5 | Endoplasmic reticulum stress contributes to beta cell apoptosis in type 2 diabetes (`25510546`) | -5.775685 |  |

### Interpretation

Labels: `terminology mismatch`, `lexical distraction`.

The terse SRL reference is not expanded in the annotated abstract, but ethanol adaptation makes the gold study a strong first-stage match. Reranking favors papers with explicit generic stress-response wording and moves the relevant ethanol-tolerance study below the cutoff.
