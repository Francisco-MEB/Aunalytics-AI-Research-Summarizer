# Google Cloud Run Deployment Guide

## Prerequisites

1. **Google Cloud Account** with billing enabled
2. **gcloud CLI** installed: https://cloud.google.com/sdk/docs/install
3. **Docker** installed locally for testing

## Setup Steps

### 1. Initialize Google Cloud Project

```powershell
# Login to Google Cloud
gcloud auth login

# Set your project ID
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable containerregistry.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

### 2. Build and Test Docker Image Locally

```powershell
# Build the Docker image
docker build -t aunalytics-rag-api .

# Test locally (make sure .env file exists)
docker run -p 8080:8080 --env-file .env aunalytics-rag-api

# Test endpoints
curl http://localhost:8080/
curl http://localhost:8080/health
```

### 3. Push to Google Container Registry

```powershell
# Configure Docker for GCR
gcloud auth configure-docker

# Tag the image for GCR
docker tag aunalytics-rag-api gcr.io/YOUR_PROJECT_ID/aunalytics-rag-api:latest

# Push to GCR
docker push gcr.io/YOUR_PROJECT_ID/aunalytics-rag-api:latest
```

### 4. Deploy to Cloud Run

```powershell
# Deploy with environment variables
gcloud run deploy aunalytics-rag-api `
  --image gcr.io/YOUR_PROJECT_ID/aunalytics-rag-api:latest `
  --platform managed `
  --region us-central1 `
  --allow-unauthenticated `
  --memory 2Gi `
  --cpu 2 `
  --timeout 300 `
  --set-env-vars "GEMINI_API_KEY=your_gemini_key" `
  --set-env-vars "user=your_db_user" `
  --set-env-vars "password=your_db_password" `
  --set-env-vars "host=your_db_host" `
  --set-env-vars "port=5432" `
  --set-env-vars "dbname=postgres"
```

**Better: Use Secret Manager for sensitive data:**

```powershell
# Create secrets (one-time setup)
echo -n "your_gemini_key" | gcloud secrets create gemini-api-key --data-file=-
echo -n "your_db_password" | gcloud secrets create db-password --data-file=-

# Deploy with secrets
gcloud run deploy aunalytics-rag-api `
  --image gcr.io/YOUR_PROJECT_ID/aunalytics-rag-api:latest `
  --platform managed `
  --region us-central1 `
  --allow-unauthenticated `
  --memory 2Gi `
  --cpu 2 `
  --timeout 300 `
  --set-secrets "GEMINI_API_KEY=gemini-api-key:latest" `
  --set-secrets "password=db-password:latest" `
  --set-env-vars "user=your_db_user,host=your_db_host,port=5432,dbname=postgres"
```

### 5. Quick Deploy Script

Create `deploy.ps1`:

```powershell
# Quick deployment script
param(
    [Parameter(Mandatory=$true)]
    [string]$ProjectId
)

Write-Host "Deploying to Google Cloud Run..." -ForegroundColor Green

# Build and push
Write-Host "Building Docker image..." -ForegroundColor Yellow
docker build -t aunalytics-rag-api .

Write-Host "Tagging for GCR..." -ForegroundColor Yellow
docker tag aunalytics-rag-api gcr.io/$ProjectId/aunalytics-rag-api:latest

Write-Host "Pushing to GCR..." -ForegroundColor Yellow
docker push gcr.io/$ProjectId/aunalytics-rag-api:latest

Write-Host "Deploying to Cloud Run..." -ForegroundColor Yellow
gcloud run deploy aunalytics-rag-api `
  --image gcr.io/$ProjectId/aunalytics-rag-api:latest `
  --platform managed `
  --region us-central1 `
  --allow-unauthenticated `
  --memory 2Gi `
  --cpu 2 `
  --timeout 300 `
  --set-secrets "GEMINI_API_KEY=gemini-api-key:latest" `
  --set-secrets "password=db-password:latest" `
  --set-env-vars "user=your_db_user,host=your_db_host,port=5432,dbname=postgres"

Write-Host "Deployment complete!" -ForegroundColor Green
```

## Testing Deployed API

```powershell
# Get the Cloud Run URL
$API_URL = gcloud run services describe aunalytics-rag-api --region us-central1 --format "value(status.url)"

# Test health endpoint
curl "$API_URL/health"

# Test upload (replace with your user_id)
curl -X POST "$API_URL/api/upload" `
  -F "file=@data/sample.txt" `
  -F "user_id=71b3dc50-fd36-49bc-856a-24cb64387b43"

# Test question
curl -X POST "$API_URL/api/question" `
  -H "Content-Type: application/json" `
  -d '{"question": "What is this about?", "user_id": "71b3dc50-fd36-49bc-856a-24cb64387b43"}'
```

## Configuration Notes

### Memory & CPU
- **2Gi memory**: Needed for sentence transformer model
- **2 CPU**: Improves embedding generation speed
- Adjust based on usage and cost

### Timeout
- **300s**: For large document uploads
- Default is 60s, increase for processing time

### Scaling
- Cloud Run auto-scales (0 to N instances)
- Set max instances if needed:
  ```
  --max-instances 10
  ```

### Database Connection
- Ensure Supabase allows connections from Cloud Run IPs
- Or use Cloud SQL Proxy for private connections

## Cost Estimation

**Cloud Run Pricing (us-central1):**
- CPU: $0.00002400/vCPU-second
- Memory: $0.00000250/GiB-second
- Requests: Free tier 2M requests/month

**Example monthly cost:**
- 100 requests/day × 30 days = 3,000 requests
- Average 5 seconds per request
- 2 vCPU, 2Gi RAM
- **Estimated: ~$5-10/month** (within free tier if low usage)

## Troubleshooting

### Container fails to start
```powershell
# Check logs
gcloud run services logs read aunalytics-rag-api --region us-central1
```

### Database connection issues
- Verify Supabase host/port/credentials
- Check firewall rules
- Test connection from Cloud Shell

### Out of memory
- Increase memory allocation
- Profile model loading

### Slow cold starts
- First request takes ~15s (model loading)
- Consider min-instances for always-warm:
  ```
  --min-instances 1
  ```
  (costs more but faster response)
