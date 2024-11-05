# Predicting If Statements in Python Code Using T5
## Project Documentation

### Why CodeSearchNet?

For gathering training data, we chose to use the CodeSearchNet dataset rather than scraping GitHub directly. This saved us considerable preprocessing effort - CodeSearchNet provides clean, well-documented Python functions right out of the box. No need to deal with parsing raw files or filtering out low-quality code.

We also went with T5-small as our base model. While CodeT5+ or CodeBERT might seem like more obvious choices for a code-related task, we found T5-small offered a good balance of performance and practicality. It's lightweight enough to train on limited compute resources but still powerful enough for our needs.

### 1. Dataset Construction

#### 1.1 Data Source
We utilized the CodeSearchNet dataset, which consists of 14 training files, 1 validation file, and 1 test file in JSONL.GZ format. The dataset was chosen for its:
- High-quality Python function collection
- Pre-processed code structure
- Built-in train/validation/test split
- Rich metadata including repository information and docstrings

#### 1.2 Data Processing Pipeline
Our data processing workflow consisted of:

1. First, we loaded the compressed files into pandas DataFrames:
```python
def jsonl_list_to_dataframe(file_list, columns):
    return pd.concat([pd.read_json(f, 
                     orient='records', 
                     compression='gzip',
                     lines=True)[columns] 
                     for f in file_list], sort=False)
```

2. Then filtered for functions containing if statements:
```python
def filter_if_statements(df):
    return df[df['code'].str.contains(r'\bif\b')]
```

Following the assignment requirements, we split the data into:
- Pre-training: 150,000 samples
- Fine-tuning total: 50,000 samples
  * Training: 40,000 samples (80%)
  * Validation: 5,000 samples (10%)
  * Testing: 5,000 samples (10%)

### 2. Model Architecture and Training

#### 2.1 Model Selection
We implemented our solution using the T5-small model architecture:
- Base model: t5-small from HuggingFace
- Tokenizer: T5Tokenizer with max_length=512
- Special tokens: Used `<extra_id_0>` for masking

#### 2.2 Training Strategy

1. **Pre-training Phase**:
   - Epochs: 3
   - Batch size: 8
   - Learning rate: 5e-5
   - Optimizer: AdamW
   - Masking rate: 15% of tokens
   - Training device: GPU
   - Results:
     * Epoch 1 Loss: 0.270
     * Epoch 2 Loss: 0.183
     * Epoch 3 Loss: 0.181

2. **Fine-tuning Phase**:
   - Epochs: 3
   - Same hyperparameters as pre-training
   - Results:
     * Epoch 1 Loss: 0.17851
     * Epoch 2 Loss: 0.17839
     * Epoch 3 Loss: 0.17839

#### 2.3 Masked Language Modeling Strategy

For pre-training, we implemented Masked Language Modeling (MLM) as our key learning approach. Here's why and how:

1. **Basic Concept**:
   - We randomly mask 15% of tokens in each code snippet
   - The model learns to predict these masked tokens
   - This forces it to understand code context and structure

2. **Implementation Details**:
```python
# During training, we mask tokens randomly
mask_idx = torch.rand(input_ids.shape).to(input_ids.device) < 0.15  
input_ids[mask_idx] = mask_token_id

# Model tries to predict the original tokens
outputs = model(input_ids=input_ids, 
               attention_mask=attention_mask, 
               labels=input_ids)
```

3. **Why MLM for Code?**
   - Code has strong structural dependencies
   - MLM helps learn both local and global patterns
   - The model must understand context to make correct predictions
   - Particularly useful for understanding if statement patterns

### 3. Implementation Details

#### 3.1 Dataset Implementation
```python
class CodeDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        code = self.data.iloc[idx]['code']
        inputs = self.tokenizer(
            code, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length
        )
        return inputs['input_ids'].squeeze(), inputs['attention_mask'].squeeze()
```

#### 3.2 Evaluation Results
- Validation Loss: 0.0754
- Test Loss: 0.0798
- Generated test output shows model can:
  * Maintain code structure
  * Preserve if statement context
  * Generate syntactically correct code

#### 3.3 Result Generation
For evaluation, we built a test harness that generates CSV files with model predictions:
```python
def generate_testset_csv(test_data, input_col, target_col, csv_filename):
    # Generates CSV with:
    # - Original input
    # - Prediction correctness
    # - Expected/predicted conditions
    # - Confidence scores
```



This project demonstrated that MLM pre-training combined with a focused fine-tuning approach can effectively learn Python code patterns. The key was a combination of careful data preparation, appropriate pre-training strategy, and targeted fine-tuning.

The code and detailed results are available in our GitHub repository: https://github.com/Sun20010423/ASS2_PTLM.git
