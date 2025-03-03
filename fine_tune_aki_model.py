import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

# Step 1: Define the 116 cases as a list of dictionaries
cases =cases = [
    {"input": "65-year-old male with hypertension and diabetes due to volume depletion.", "output": "AKI likely due to volume depletion in a patient with hypertension and diabetes. Recommend IV fluids and monitor renal function."},
    {"input": "70-year-old female post-cardiac catheterization due to contrast-induced nephropathy.", "output": "AKI likely due to contrast-induced nephropathy. Recommend hydration and monitor renal function."},
    {"input": "55-year-old male with sepsis.", "output": "AKI likely due to sepsis. Recommend broad-spectrum antibiotics and fluid resuscitation."},
    {"input": "60-year-old male with obstructive uropathy.", "output": "AKI likely due to obstructive uropathy. Recommend relief of obstruction and monitor renal function."},
    {"input": "45-year-old male with suspected acute glomerulonephritis.", "output": "AKI likely due to acute glomerulonephritis. Recommend further workup including urine analysis and renal biopsy."},
    {"input": "68-year-old male post-coronary artery bypass grafting (CABG).", "output": "AKI likely due to post-surgical complications. Monitor renal function and optimize hemodynamics."},
    {"input": "72-year-old female with decompensated heart failure (cardio-renal syndrome).", "output": "AKI likely due to cardio-renal syndrome. Optimize heart failure management and monitor renal function."},
    {"input": "58-year-old male on multiple nephrotoxic medications.", "output": "AKI likely due to nephrotoxic medications. Review and adjust medications, and monitor renal function."},
    {"input": "70-year-old female with CKD stage 3 and volume depletion.", "output": "AKI on CKD stage 3 likely due to volume depletion. Recommend IV fluids and monitor renal function."},
    {"input": "65-year-old male with CKD stage 4 and obstructive uropathy.", "output": "AKI on CKD stage 4 likely due to obstructive uropathy. Recommend relief of obstruction and monitor renal function."},
    {"input": "60-year-old male with CKD stage 5 and sepsis.", "output": "AKI on CKD stage 5 likely due to sepsis. Recommend broad-spectrum antibiotics and fluid resuscitation."},
    {"input": "55-year-old female with CKD stage 3 on nephrotoxic medications.", "output": "AKI on CKD stage 3 likely due to nephrotoxic medications. Review and adjust medications, and monitor renal function."},
    {"input": "30-year-old male with suspected infection-related glomerulonephritis (GN) secondary to tricuspid valve endocarditis.", "output": "AKI likely due to infection-related GN. Recommend further workup including blood cultures and echocardiogram."},
    {"input": "35-year-old female with dengue fever.", "output": "AKI likely due to dengue fever. Recommend supportive care and monitor renal function."},
    {"input": "65-year-old male with myeloma cast nephropathy.", "output": "AKI likely due to myeloma cast nephropathy. Recommend further workup including SPEP and free light chains."},
    {"input": "40-year-old male with Fabry disease.", "output": "AKI likely due to Fabry disease. Recommend enzyme replacement therapy and monitor renal function."},
    {"input": "25-year-old male with Alport syndrome.", "output": "AKI likely due to Alport syndrome. Recommend genetic counseling and monitor renal function."},
    {"input": "28-year-old female with malaria.", "output": "AKI likely due to malaria. Recommend antimalarial therapy and monitor renal function."},
    {"input": "50-year-old male with hepatorenal syndrome (HRS) type 1.", "output": "AKI likely due to HRS type 1. Recommend vasopressor therapy and consider liver transplant evaluation."},
    {"input": "40-year-old male with septic shock, contrast exposure, and NSAID use.", "output": "AKI likely due to multifactorial causes including septic shock, contrast exposure, and NSAID use. Recommend discontinuation of NSAIDs, hydration, and broad-spectrum antibiotics."},
    {"input": "45-year-old male with suspected acute interstitial nephritis (AIN) following antibiotic therapy.", "output": "AKI likely due to AIN. Recommend discontinuation of offending antibiotic and consider steroid therapy."},
    {"input": "74-year-old male with CKD stage III and community-acquired pneumonia (CAP).", "output": "AKI on CKD stage III likely due to CAP. Recommend antibiotics and monitor renal function."},
    {"input": "35-year-old male following a snake bite.", "output": "AKI likely due to snake bite. Recommend antivenom therapy and monitor renal function."},
    {"input": "60-year-old male with amyloidosis.", "output": "AKI likely due to amyloidosis. Recommend further workup including serum free light chains and renal biopsy."},
    {"input": "50-year-old female with Amanita phalloides (death cap mushroom) poisoning.", "output": "AKI likely due to Amanita phalloides poisoning. Recommend supportive care and consider liver transplant evaluation."},
    {"input": "55-year-old male post-cardiac arrest on hypothermia protocol.", "output": "AKI likely due to post-cardiac arrest syndrome. Monitor renal function and optimize hemodynamics."},
    {"input": "40-year-old male with severe diabetic ketoacidosis (DKA).", "output": "AKI likely due to DKA. Recommend insulin therapy, fluid resuscitation, and monitor renal function."},
    {"input": "85-year-old male with severe bilateral hydronephrosis.", "output": "AKI likely due to obstructive uropathy. Recommend relief of obstruction and monitor renal function."},
    {"input": "50-year-old female with malignant hypertension.", "output": "AKI likely due to malignant hypertension. Recommend aggressive blood pressure control and monitor renal function."},
    {"input": "65-year-old male after starting ACEI/ARB and SGLT-2 inhibitor.", "output": "AKI likely due to medication-induced renal injury. Recommend discontinuation of ACEI/ARB and SGLT-2 inhibitor, and monitor renal function."},
    {"input": "80-year-old female with failure to thrive.", "output": "AKI likely due to chronic illness and malnutrition. Recommend nutritional support and monitor renal function."},
    {"input": "85-year-old male with severe benign prostatic hyperplasia (BPH).", "output": "AKI likely due to obstructive uropathy. Recommend relief of obstruction and monitor renal function."},
    {"input": "60-year-old female with cancer and chemotherapy-induced nausea/vomiting and diarrhea.", "output": "AKI likely due to volume depletion from chemotherapy-induced nausea/vomiting and diarrhea. Recommend IV fluids and antiemetics."},
    {"input": "55-year-old male with high ostomy output post-colostomy.", "output": "AKI likely due to volume depletion from high ostomy output. Recommend IV fluids and monitor renal function."},
    {"input": "68-year-old male with cardio-renal syndrome and SGLT-2 inhibitor, Entresto, and spironolactone.", "output": "AKI likely due to cardio-renal syndrome and medication effects. Recommend review and adjustment of medications, and monitor renal function."},
    {"input": "72-year-old female after starting Entresto.", "output": "AKI likely due to medication-induced renal injury. Recommend discontinuation of Entresto and monitor renal function."},
    {"input": "50-year-old male with bile cast nephropathy.", "output": "AKI likely due to bile cast nephropathy. Recommend further workup including liver function tests and renal biopsy."},
    {"input": "35-year-old male with rhabdomyolysis.", "output": "AKI likely due to rhabdomyolysis. Recommend aggressive IV fluids and monitor renal function."},
    {"input": "60-year-old female on acyclovir.", "output": "AKI likely due to acyclovir-induced nephrotoxicity. Recommend discontinuation of acyclovir and monitor renal function."},
    {"input": "55-year-old male on Bactrim.", "output": "AKI likely due to Bactrim-induced nephrotoxicity. Recommend discontinuation of Bactrim and monitor renal function."},
    {"input": "45-year-old male with ANCA vasculitis.", "output": "AKI likely due to ANCA vasculitis. Recommend further workup including ANCA profile and consider immunosuppressive therapy."},
    {"input": "30-year-old female with lupus nephritis.", "output": "AKI likely due to lupus nephritis. Recommend further workup including ANA, anti-dsDNA, and consider immunosuppressive therapy."},
    {"input": "50-year-old male with anti-GBM disease.", "output": "AKI likely due to anti-GBM disease. Recommend further workup including anti-GBM antibodies and consider plasmapheresis."},
    {"input": "65-year-old male with COVID-19 pneumonitis.", "output": "AKI likely due to COVID-19 pneumonitis. Recommend supportive care and monitor renal function."},
    {"input": "40-year-old male with pancreatitis.", "output": "AKI likely due to pancreatitis. Recommend aggressive IV fluids and monitor renal function."},
    {"input": "55-year-old female with small bowel obstruction (SBO).", "output": "AKI likely due to volume depletion from SBO. Recommend IV fluids and surgical consultation."},
    {"input": "45-year-old male with nephrotic syndrome (MCD) and anasarca.", "output": "AKI likely due to nephrotic syndrome. Recommend further workup including urine protein-to-creatinine ratio and consider steroid therapy."},
    {"input": "50-year-old male post-kidney transplant with calcineurin inhibitor (CNI) toxicity.", "output": "AKI likely due to CNI toxicity. Recommend adjustment of CNI dose and monitor renal function."},
    {"input": "55-year-old female post-kidney transplant with acute rejection.", "output": "AKI likely due to acute rejection. Recommend further workup including renal biopsy and consider immunosuppressive therapy."},
    {"input": "70-year-old male with atrial fibrillation.", "output": "AKI likely due to hemodynamic instability from atrial fibrillation. Recommend rate control and monitor renal function."},
    {"input": "40-year-old male with infection-related glomerulonephritis (GN).", "output": "AKI likely due to infection-related GN. Recommend further workup including complement levels and ANCA profile."},
    {"input": "35-year-old male with HIV immune complex-mediated disease.", "output": "AKI likely due to HIV immune complex-mediated disease. Recommend further workup including HIV viral load and consider immunosuppressive therapy."},
    {"input": "50-year-old female following weight loss medication use (Ozempic, Mounjaro).", "output": "AKI likely due to medication-induced renal injury. Recommend discontinuation of weight loss medications and monitor renal function."},
    {"input": "30-year-old male with systemic IgA vasculitis.", "output": "AKI likely due to IgA vasculitis. Recommend further workup including renal biopsy and consider immunosuppressive therapy."},
    {"input": "30-year-old pregnant female with preeclampsia.", "output": "AKI likely due to preeclampsia. Recommend delivery of the fetus and monitor renal function."},
    {"input": "45-year-old female with thrombotic microangiopathy (TMA).", "output": "AKI likely due to TMA. Recommend further workup including ADAMTS-13 level and consider plasmapheresis."},
    {"input": "55-year-old female with lithium toxicity (medically managed).", "output": "AKI likely due to lithium toxicity. Recommend discontinuation of lithium and monitor renal function."},
    {"input": "60-year-old male with severe lithium toxicity (managed with dialysis).", "output": "AKI likely due to severe lithium toxicity. Recommend dialysis and monitor renal function."},
    {"input": "50-year-old male with hepatorenal syndrome (HRS).", "output": "AKI likely due to HRS. Recommend vasopressor therapy and consider liver transplant evaluation."},
    {"input": "65-year-old female with worsening ascites.", "output": "AKI likely due to hepatorenal syndrome. Recommend paracentesis and monitor renal function."},
    {"input": "35-year-old male with rhabdomyolysis (managed with fluids).", "output": "AKI likely due to rhabdomyolysis. Recommend aggressive IV fluids and monitor renal function."},
    {"input": "40-year-old male with severe rhabdomyolysis (requiring dialysis).", "output": "AKI likely due to severe rhabdomyolysis. Recommend dialysis and monitor renal function."},
    {"input": "45-year-old male with hepatorenal syndrome (HRS) awaiting liver transplant.", "output": "AKI likely due to HRS. Recommend vasopressor therapy and monitor renal function."},
    {"input": "50-year-old female with septic shock.", "output": "AKI likely due to septic shock. Recommend broad-spectrum antibiotics and vasopressor therapy."},
    {"input": "65-year-old male with cardiogenic shock.", "output": "AKI likely due to cardiogenic shock. Recommend inotropic support and monitor renal function."},
    {"input": "45-year-old male with hypovolemic shock requiring CRRT.", "output": "AKI likely due to hypovolemic shock. Recommend CRRT and monitor renal function."},
    {"input": "30-year-old male with ethylene glycol poisoning.", "output": "AKI likely due to ethylene glycol poisoning. Recommend fomepizole and dialysis."},
    {"input": "40-year-old male with tumor lysis syndrome.", "output": "AKI likely due to tumor lysis syndrome. Recommend rasburicase and aggressive IV fluids."},
    {"input": "55-year-old female with acute interstitial nephritis (AIN) due to antibiotics.", "output": "AKI likely due to AIN. Recommend discontinuation of offending antibiotic and consider steroid therapy."},
    {"input": "50-year-old female on cisplatin for lung cancer.", "output": "AKI likely due to cisplatin-induced nephrotoxicity. Recommend discontinuation of cisplatin and monitor renal function."},
    {"input": "60-year-old male on bevacizumab for colorectal cancer.", "output": "AKI likely due to bevacizumab-induced nephrotoxicity. Recommend discontinuation of bevacizumab and monitor renal function."},
    {"input": "45-year-old male post-nephrectomy.", "output": "AKI likely due to post-surgical complications. Monitor renal function and optimize hemodynamics."},
    {"input": "68-year-old male post-CABG.", "output": "AKI likely due to post-surgical complications. Monitor renal function and optimize hemodynamics."},
    {"input": "72-year-old female with decompensated heart failure (cardio-renal syndrome).", "output": "AKI likely due to cardio-renal syndrome. Optimize heart failure management and monitor renal function."},
    {"input": "58-year-old male on multiple nephrotoxic medications.", "output": "AKI likely due to nephrotoxic medications. Review and adjust medications, and monitor renal function."},
    {"input": "70-year-old female with CKD stage 3 and volume depletion.", "output": "AKI on CKD stage 3 likely due to volume depletion. Recommend IV fluids and monitor renal function."},
    {"input": "65-year-old male with CKD stage 4 and obstructive uropathy.", "output": "AKI on CKD stage 4 likely due to obstructive uropathy. Recommend relief of obstruction and monitor renal function."},
    {"input": "60-year-old male with CKD stage 5 and sepsis.", "output": "AKI on CKD stage 5 likely due to sepsis. Recommend broad-spectrum antibiotics and fluid resuscitation."},
    {"input": "55-year-old female with CKD stage 3 on nephrotoxic medications.", "output": "AKI on CKD stage 3 likely due to nephrotoxic medications. Review and adjust medications, and monitor renal function."},
    {"input": "50-year-old male with hypercalcemia (workup: SPEP, free light chain ratio, PTH, PTHrP, Vitamin D, calcitriol, CT scan, ACE level).", "output": "AKI likely due to hypercalcemia. Recommend further workup including SPEP, free light chain ratio, and PTH levels."},
    {"input": "AKI concerning for RPGN (workup: ANA, ANCA profile, MPO, PR-3, C3, C4, SPEP, free light chain ratio, PLA2R, anti-GBM).", "output": "AKI likely due to RPGN. Recommend further workup including ANCA profile, complement levels, and renal biopsy."},
    {"input": "AKI with malignant hypertension (workup: renal duplex).", "output": "AKI likely due to malignant hypertension. Recommend aggressive blood pressure control and renal duplex."},
    {"input": "AKI with suspected infection-related GN (workup: complement levels, ANCA profile, quantify proteinuria).", "output": "AKI likely due to infection-related GN. Recommend further workup including complement levels and ANCA profile."},
    {"input": "AKI due to pre-renal azotemia (workup: urine sodium, chloride, FeNa, renal US, medication review).", "output": "AKI likely due to pre-renal azotemia. Recommend IV fluids and monitor renal function."},
    {"input": "AKI with severe hyperkalemia refractory to medical therapy (initiated on dialysis).", "output": "AKI with severe hyperkalemia. Recommend dialysis and monitor renal function."},
    {"input": "AKI with minimal UOP, severe hypotension, and worsening lactic acidosis (initiated on CRRT).", "output": "AKI with severe hemodynamic instability. Recommend CRRT and monitor renal function."},
    {"input": "AKI with severe uremic encephalopathy (initiated on dialysis).", "output": "AKI with severe uremic encephalopathy. Recommend dialysis and monitor renal function."},
    {"input": "AKI with toxic ingestion requiring urgent dialysis.", "output": "AKI likely due to toxic ingestion. Recommend dialysis and monitor renal function."},
    {"input": "AKI with severe rhabdomyolysis and refractory hyperkalemia (initiated on dialysis).", "output": "AKI likely due to severe rhabdomyolysis. Recommend dialysis and monitor renal function."},
    {"input": "Tumor lysis syndrome refractory to rasburicase and bicarbonate (initiated on dialysis).", "output": "AKI likely due to tumor lysis syndrome. Recommend dialysis and monitor renal function."},
    {"input": "Refractory volume overload in advanced CHF (initiated on CRRT for volume optimization).", "output": "AKI likely due to advanced CHF. Recommend CRRT and monitor renal function."},
    {"input": "AKI with pulmonary-renal syndrome and diffuse alveolar hemorrhage (initiated on plasmapheresis).", "output": "AKI likely due to pulmonary-renal syndrome. Recommend plasmapheresis and monitor renal function."},
    {"input": "AKI in a 12-year-old male following sore throat 2 weeks prior, presenting with hematuria and AKI (post-infectious GN).", "output": "AKI likely due to post-infectious GN. Recommend further workup including ASO titers and monitor renal function."},
    {"input": "AKI in a 10-year-old female with diarrhea after eating a hamburger, concerning for HUS (Shiga toxin or E. coli O157:H7).", "output": "AKI likely due to HUS. Recommend further workup including Shiga toxin testing and monitor renal function."},
    {"input": "AKI in a 6-year-old male with abrupt onset swelling, concerning for minimal change disease (MCD).", "output": "AKI likely due to MCD. Recommend further workup including urine protein-to-creatinine ratio and consider steroid therapy."},
    {"input": "AKI in a 35-year-old female with thrombotic thrombocytopenic purpura (TTP) (workup: ADAMTS-13 level, plasmapheresis, caplacizumab for refractory TTP).", "output": "AKI likely due to TTP. Recommend plasmapheresis and monitor ADAMTS-13 levels."},
    {"input": "AKI in a 40-year-old male with skin gangrene and excessive vegetable intake, concerning for oxalate nephropathy.", "output": "AKI likely due to oxalate nephropathy. Recommend further workup including urine oxalate levels and monitor renal function."},
    {"input": "AKI in a 50-year-old female following 7 days of aminoglycoside use (ATN).", "output": "AKI likely due to aminoglycoside-induced ATN. Recommend discontinuation of aminoglycosides and monitor renal function."},
    {"input": "AKI in a 55-year-old male following vancomycin use with high trough levels (ATN).", "output": "AKI likely due to vancomycin-induced ATN. Recommend discontinuation of vancomycin and monitor renal function."},
    {"input": "AKI with worsening renal function due to HRS in a patient not eligible for liver transplant (dialysis not helpful).", "output": "AKI likely due to HRS. Recommend supportive care and monitor renal function."},
    {"input": "AKI with refractory septic shock on 3 pressors (RRT not helpful).", "output": "AKI likely due to refractory septic shock. Recommend supportive care and monitor renal function."},
    {"input": "AKI with normal baseline creatinine, now on dialysis (suspected ATN, excellent prognosis for renal recovery in 90 days).", "output": "AKI likely due to ATN. Recommend dialysis and monitor renal function."},
    {"input": "AKI where dialysis is not indicated due to poor prognosis (goals of care discussion, conservative management of hyperkalemia, metabolic acidosis, and volume overload).", "output": "AKI with poor prognosis. Recommend goals of care discussion and conservative management."},
    {"input": "AKI in a patient on hypothermia protocol with minimal neurological activity (dialysis not indicated).", "output": "AKI likely due to hypothermia protocol. Recommend supportive care and monitor renal function."},
]

# Step 2: Convert the list of dictionaries into a pandas DataFrame
data = pd.DataFrame(cases)

# Step 3: Clean the text (if needed)
def clean_text(text):
    text = text.strip()  # Remove leading/trailing spaces
    text = text.replace("\n", " ")  # Replace newlines with spaces
    return text

data["input"] = data["input"].apply(clean_text)
data["output"] = data["output"].apply(clean_text)

# Step 4: Tokenize the dataset
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Add a padding token if not already present
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

def tokenize_function(examples):
    return tokenizer(examples["input"], examples["output"], padding="max_length", truncation=True, max_length=512)

# Convert DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(data)

# Tokenize the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Step 5: Fine-tune the GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Resize the token embeddings to account for the new padding token
model.resize_token_embeddings(len(tokenizer))

# Define training arguments
training_args = TrainingArguments(
    output_dir="./aki_gpt2_finetuned",  # Directory to save the model
    per_device_train_batch_size=4,     # Batch size
    num_train_epochs=3,                # Number of epochs
    logging_dir="./logs",              # Directory for logs
    save_steps=500,                    # Save model every 500 steps
    save_total_limit=2,                # Keep only the last 2 models
    evaluation_strategy="no",          # No evaluation during training
    logging_steps=100,                 # Log every 100 steps
    learning_rate=5e-5,                # Learning rate
    weight_decay=0.01,                 # Weight decay
    warmup_steps=100,                  # Warmup steps
    fp16=True,                         # Use mixed precision for faster training
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Fine-tune the model
trainer.train()

# Step 6: Save the fine-tuned model and tokenizer
model.save_pretrained("./aki_gpt2_finetuned")
tokenizer.save_pretrained("./aki_gpt2_finetuned")

print("Fine-tuning complete! Model saved to './aki_gpt2_finetuned'.")
