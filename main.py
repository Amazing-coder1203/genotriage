import json
import logging
import math
import numpy as np
import pandas as pd
import requests
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any
from fastapi import UploadFile, File, BackgroundTasks
from fastapi.responses import Response, JSONResponse
import io
import uuid
import time

app = FastAPI(title="GenePath AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
try:
    # Load model directly from the root directory
    model = joblib.load("genepath_model.pkl")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    model = None

class VariantInput(BaseModel):
    chromosome: str
    position: int
    ref_allele: str
    alt_allele: str

def encode_chromosome(chrom: str) -> int:
    c = str(chrom).upper().replace('CHR', '')
    if c == 'X':
        return 23
    elif c == 'Y':
        return 24
    elif c in ('M', 'MT'):
        return 25
    try:
        return int(c)
    except:
        return 0 # Fallback

def get_nested(data: Any, path: str, default: float = -1.0) -> float:
    keys = path.split('.')
    def _get(obj, keys):
        if not keys:
            return obj
        k = keys[0]
        if isinstance(obj, dict):
            return _get(obj.get(k), keys[1:])
        elif isinstance(obj, list):
            res = [_get(item, keys) for item in obj]
            res = [x for x in res if x is not None]
            return res if res else None
        return None
        
    val = _get(data, keys)
    def flatten(lst):
        if not isinstance(lst, list):
            return [lst]
        res = []
        for i in lst:
            res.extend(flatten(i))
        return res
        
    if val is None:
        return default
        
    flat = flatten(val)
    nums = []
    for x in flat:
        try:
            nums.append(float(x))
        except:
            pass
    if nums:
        return max(nums)
    return default

def extract_deep_field(obj, path_string):
    """
    Safely traverses a JSON tree along 'path.to.key'.
    Automatically unpacks lists if a step in the path contains an array.
    """
    if not obj or not path_string: return None
    parts = path_string.split('.')
    current = obj
    for part in parts:
        if isinstance(current, list):
            current = current[0] if len(current) > 0 else {}
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current

def format_pharmgkb_report(insights):
    if not insights:
        return ""
        
    blocks = []
    for insight in insights:
        header = insight.get("annotation_name", "")
        evidence = insight.get("evidence_level", "")
        drugs = " ".join(insight.get("affected_drugs", []))
        diseases = ", ".join(insight.get("associated_diseases", []))
        
        block = f"{header}\nEvidence Level: {evidence}\nTarget Drugs:\n{drugs}\nAssociated Diseases:\n{diseases}\nCLINICAL DIRECTION\n"
        
        sentences = insight.get("guideline_sentences", [])
        for s in sentences:
            allele = s.get("allele_variant", "")
            direction = s.get("clinical_direction", "")
            block += f"Allele {allele}: {direction}\n"
            
        blocks.append(block.strip())
        
    return "\n\n".join(blocks)

batch_jobs = {}

@app.post("/predict")
def predict_variant(variant: VariantInput):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    # Format HGVS
    hgvs = f"chr{variant.chromosome}:g.{variant.position}{variant.ref_allele}>{variant.alt_allele}"
    
    # API Call
    url = "https://myvariant.info/v1/variant?assembly=hg38"
    payload = {
        "ids": [hgvs],
        "fields": "cadd.phred,dbnsfp.cadd.phred,dbnsfp.sift,dbnsfp.polyphen2,dbnsfp.alphamissense,gnomad_exome.af.af,clinvar,dbnsfp.clinvar.trait,pharmgkb,dbsnp"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"MyVariant.info error: {str(e)}")
        # If MyVariant fails, return baseline with null insights
        return {
            "prediction": "Unknown",
            "confidence": 0.0,
            "probability_pathogenic": 0.0,
            "scores_used": {},
            "pharmgkb_insights": None,
            "hgvs": hgvs
        }
        
    if not data or not isinstance(data, list):
        raise HTTPException(status_code=404, detail="Unexpected response from myvariant.info")
        
    res = data[0]
    if "notfound" in res and res["notfound"]:
        raise HTTPException(status_code=404, detail="Variant not found in myvariant.info database")
        
    # Extract Features
    sift = get_nested(res, "dbnsfp.sift.converted_rankscore", -1.0)
    polyphen = get_nested(res, "dbnsfp.polyphen2.hdiv.score", -1.0)
    alphamissense = get_nested(res, "dbnsfp.alphamissense.score", -1.0)
    
    gnomad_e = get_nested(res, "gnomad_exome.af.af", None)
    gnomad_g = get_nested(res, "gnomad_genome.af.af", None)
    
    gnomad = gnomad_e if gnomad_e is not None else gnomad_g
    if gnomad is None:
        gnomad = 0.0
        
    gnomad_af_log = np.log10(gnomad + 1e-8)
    chrom_encoded = encode_chromosome(variant.chromosome)
    gene_freq = 286
    num_submitters = 1
    
    # Feature ordering as per genepath_features.json
    # ["CHROM_ENCODED", "GENE_FREQ", "NUM_SUBMITTERS", "SIFT_SCORE", "POLYPHEN_SCORE", "ALPHAMISSENSE_SCORE", "GNOMAD_AF_LOG"]
    features = pd.DataFrame([{
        "CHROM_ENCODED": chrom_encoded,
        "GENE_FREQ": gene_freq,
        "NUM_SUBMITTERS": num_submitters,
        "SIFT_SCORE": sift,
        "POLYPHEN_SCORE": polyphen,
        "ALPHAMISSENSE_SCORE": alphamissense,
        "GNOMAD_AF_LOG": gnomad_af_log
    }])
    
    # Predict
    try:
        proba = model.predict_proba(features)[0]
        prob_pathogenic = proba[1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")
        
    # Custom Threshold
    threshold = 0.636
    prediction = "Pathogenic" if prob_pathogenic >= threshold else "Benign"
    
    # PHARMGKB PIPELINE
    pharmgkb_insights = None
    
    gene_symbol = (extract_deep_field(res, "clinvar.gene.symbol") or 
                   extract_deep_field(res, "dbsnp.gene.symbol"))
    
    target_rsid = (extract_deep_field(res, "dbsnp.rsid") or 
                   extract_deep_field(res, "clinvar.rsid"))
                   
    if target_rsid:
        target_rsid = str(target_rsid).strip().lower()
        if not target_rsid.startswith('rs'):
            target_rsid = f"rs{target_rsid}"
    else:
        target_rsid = None
                   
    if gene_symbol and target_rsid:
        try:
            pharmgkb_url = "https://api.pharmgkb.org/v1/data/clinicalAnnotation"
            pgkb_response = requests.get(pharmgkb_url, params={
                "location.genes.symbol": gene_symbol,
                "view": "base"
            }, timeout=5)
            
            if pgkb_response.status_code == 404:
                annotations_list = []
            elif pgkb_response.status_code == 200:
                raw_data = pgkb_response.json()
                annotations_list = raw_data if isinstance(raw_data, list) else raw_data.get("data", [])
            else:
                annotations_list = []
                
            filtered_insights = []
            for annotation in annotations_list:
                if not isinstance(annotation, dict): continue
                
                location = annotation.get("location", {}) or {}
                rsid_val = location.get("rsid")
                
                # Normalize PharmGKB primary rsid
                rsid_val_norm = str(rsid_val).strip().lower() if rsid_val else ""
                if rsid_val_norm and not rsid_val_norm.startswith('rs'):
                    rsid_val_norm = f"rs{rsid_val_norm}"
                    
                variants = location.get("variants", []) or []
                if not isinstance(variants, list): 
                    variants = [variants]
                    
                # Normalize nested variant array symbols safely
                variant_symbols_norm = []
                for v in variants:
                    if isinstance(v, dict) and v.get("symbol"):
                        sym = str(v.get("symbol")).strip().lower()
                        if not sym.startswith('rs'):
                            sym = f"rs{sym}"
                        variant_symbols_norm.append(sym)
                        
                # Execute cross-reference validation check
                if (rsid_val_norm == target_rsid) or (target_rsid in variant_symbols_norm):
                    clean_entry = {
                        "annotation_name": annotation.get("name"),
                        "evidence_level": annotation.get("levelOfEvidence", {}).get("term", "N/A"),
                        "affected_drugs": [chemical.get("name") for chemical in annotation.get("relatedChemicals", []) if isinstance(chemical, dict)],
                        "associated_diseases": [disease.get("name") for disease in annotation.get("relatedDiseases", []) if isinstance(disease, dict)],
                        "guideline_sentences": [
                            {
                                "allele_variant": pheno.get("allele"),
                                "clinical_direction": pheno.get("phenotype")
                            }
                            for pheno in annotation.get("allelePhenotypes", []) if isinstance(pheno, dict) and pheno.get("phenotype")
                        ]
                    }
                    filtered_insights.append(clean_entry)
            
            pharmgkb_insights = filtered_insights
        except Exception as e:
            print(f"PharmGKB API error: {str(e)}")
            pharmgkb_insights = None
            
    return {
        "prediction": prediction,
        "confidence": float(prob_pathogenic * 100) if prediction == "Pathogenic" else float(proba[0] * 100),
        "probability_pathogenic": float(prob_pathogenic),
        "scores_used": {
            "SIFT_SCORE": sift,
            "POLYPHEN_SCORE": polyphen,
            "ALPHAMISSENSE_SCORE": alphamissense,
            "GNOMAD_AF": gnomad,
            "GNOMAD_AF_LOG": float(gnomad_af_log),
            "CHROM_ENCODED": chrom_encoded
        },
        "pharmgkb_insights": pharmgkb_insights,
        "hgvs": hgvs
    }

def process_batch_background(job_id, df):
    try:
        hgvs_ids = []
        for _, row in df.iterrows():
            hgvs = f"chr{row['chromosome']}:g.{row['position']}{row['ref_allele']}>{row['alt_allele']}"
            hgvs_ids.append(hgvs)
            
        chunk_size = 1000
        all_responses = {}
        url = "https://myvariant.info/v1/variant?assembly=hg38"
        
        for i in range(0, len(hgvs_ids), chunk_size):
            chunk = hgvs_ids[i:i+chunk_size]
            payload = {
                "ids": chunk,
                "fields": "cadd.phred,dbnsfp.cadd.phred,dbnsfp.sift,dbnsfp.polyphen2,dbnsfp.alphamissense,gnomad_exome.af.af,clinvar,dbsnp"
            }
            try:
                response = requests.post(url, json=payload, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    for item in data:
                        if "query" in item:
                            all_responses[item["query"]] = item
            except Exception as e:
                logging.error(f"Error querying myvariant.info in batch: {e}")
                
        features_list = []
        pharmgkb_reports = []
        rsids_list = []
        
        for idx, hgvs in enumerate(hgvs_ids):
            res = all_responses.get(hgvs, {})
            
            sift = get_nested(res, "dbnsfp.sift.converted_rankscore", -1.0)
            polyphen = get_nested(res, "dbnsfp.polyphen2.hdiv.score", -1.0)
            alphamissense = get_nested(res, "dbnsfp.alphamissense.score", -1.0)
            
            gnomad_e = get_nested(res, "gnomad_exome.af.af", None)
            gnomad_g = get_nested(res, "gnomad_genome.af.af", None)
            gnomad = gnomad_e if gnomad_e is not None else gnomad_g
            if gnomad is None:
                gnomad = 0.0
                
            gnomad_af_log = np.log10(gnomad + 1e-8)
            
            row = df.iloc[idx]
            chrom_encoded = encode_chromosome(row['chromosome'])
            
            features_list.append({
                "CHROM_ENCODED": chrom_encoded,
                "GENE_FREQ": 286,
                "NUM_SUBMITTERS": 1,
                "SIFT_SCORE": sift,
                "POLYPHEN_SCORE": polyphen,
                "ALPHAMISSENSE_SCORE": alphamissense,
                "GNOMAD_AF_LOG": gnomad_af_log
            })
            
            gene_symbol = (extract_deep_field(res, "clinvar.gene.symbol") or 
                           extract_deep_field(res, "dbsnp.gene.symbol"))
            
            target_rsid = (extract_deep_field(res, "dbsnp.rsid") or 
                           extract_deep_field(res, "clinvar.rsid"))
                           
            if target_rsid:
                target_rsid = str(target_rsid).strip().lower()
                if not target_rsid.startswith('rs'):
                    target_rsid = f"rs{target_rsid}"
            else:
                target_rsid = None
                
            rsids_list.append(target_rsid if target_rsid else "")
                           
            insights = None
            if gene_symbol and target_rsid:
                try:
                    pharmgkb_url = "https://api.pharmgkb.org/v1/data/clinicalAnnotation"
                    pgkb_response = requests.get(pharmgkb_url, params={
                        "location.genes.symbol": gene_symbol,
                        "view": "base"
                    }, timeout=5)
                    
                    if pgkb_response.status_code == 200:
                        raw_data = pgkb_response.json()
                        annotations_list = raw_data if isinstance(raw_data, list) else raw_data.get("data", [])
                        
                        filtered_insights = []
                        for annotation in annotations_list:
                            if not isinstance(annotation, dict): continue
                            
                            location = annotation.get("location", {}) or {}
                            rsid_val = location.get("rsid")
                            
                            rsid_val_norm = str(rsid_val).strip().lower() if rsid_val else ""
                            if rsid_val_norm and not rsid_val_norm.startswith('rs'):
                                rsid_val_norm = f"rs{rsid_val_norm}"
                                
                            variants = location.get("variants", []) or []
                            if not isinstance(variants, list): 
                                variants = [variants]
                                
                            variant_symbols_norm = []
                            for v in variants:
                                if isinstance(v, dict) and v.get("symbol"):
                                    sym = str(v.get("symbol")).strip().lower()
                                    if not sym.startswith('rs'):
                                        sym = f"rs{sym}"
                                    variant_symbols_norm.append(sym)
                                    
                            if (rsid_val_norm == target_rsid) or (target_rsid in variant_symbols_norm):
                                clean_entry = {
                                    "annotation_name": annotation.get("name"),
                                    "evidence_level": annotation.get("levelOfEvidence", {}).get("term", "N/A"),
                                    "affected_drugs": [chemical.get("name") for chemical in annotation.get("relatedChemicals", []) if isinstance(chemical, dict)],
                                    "associated_diseases": [disease.get("name") for disease in annotation.get("relatedDiseases", []) if isinstance(disease, dict)],
                                    "guideline_sentences": [
                                        {
                                            "allele_variant": pheno.get("allele"),
                                            "clinical_direction": pheno.get("phenotype")
                                        }
                                        for pheno in annotation.get("allelePhenotypes", []) if isinstance(pheno, dict) and pheno.get("phenotype")
                                    ]
                                }
                                filtered_insights.append(clean_entry)
                        
                        if filtered_insights:
                            insights = filtered_insights
                except Exception as e:
                    logging.error(f"PharmGKB API error in batch: {e}")
                    
                time.sleep(0.2) # Rate limit delay
                
            report_str = format_pharmgkb_report(insights) if insights else ""
            pharmgkb_reports.append(report_str)
            
            # Update job progress
            batch_jobs[job_id]["current"] = idx + 1
            
        features_df = pd.DataFrame(features_list)
        
        if not features_df.empty:
            probas = model.predict_proba(features_df)
            threshold = 0.636
            predictions = ["Pathogenic" if p[1] >= threshold else "Benign" for p in probas]
            confidences = [round(float(p[1]*100), 2) if p[1] >= threshold else round(float(p[0]*100), 2) for p in probas]
            df["Prediction"] = predictions
            df["Confidence_Pct"] = confidences
            df["SIFT_Score"] = features_df["SIFT_SCORE"]
            df["PolyPhen_Score"] = features_df["POLYPHEN_SCORE"]
            df["AlphaMissense_Score"] = features_df["ALPHAMISSENSE_SCORE"]
            df["gnomAD_AF_Log"] = features_df["GNOMAD_AF_LOG"]
            
        df["rsid"] = rsids_list
        df["PharmGKB_Report"] = pharmgkb_reports
                
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        batch_jobs[job_id]["csv_data"] = output.getvalue()
        batch_jobs[job_id]["status"] = "completed"
        
    except Exception as e:
        batch_jobs[job_id]["status"] = "error"
        batch_jobs[job_id]["error"] = str(e)
        logging.error(f"Batch processing failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")
        
    required_cols = ["chromosome", "position", "ref_allele", "alt_allele"]
    for col in required_cols:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Missing required column: {col}")
            
    job_id = str(uuid.uuid4())
    total_rows = len(df)
    
    batch_jobs[job_id] = {
        "status": "processing",
        "current": 0,
        "total": total_rows,
        "csv_data": None,
        "error": None
    }
    
    background_tasks.add_task(process_batch_background, job_id, df)
    
    return JSONResponse(content={"job_id": job_id, "total": total_rows})

@app.get("/predict/batch/status/{job_id}")
async def get_batch_status(job_id: str):
    if job_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = batch_jobs[job_id]
    
    if job["status"] == "completed":
        return JSONResponse(content={
            "status": "completed",
            "current": job["current"],
            "total": job["total"],
            "csv_data": job["csv_data"]
        })
        
    return JSONResponse(content={
        "status": job["status"],
        "current": job["current"],
        "total": job["total"],
        "error": job["error"]
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
