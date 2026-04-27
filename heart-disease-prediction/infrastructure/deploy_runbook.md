# AWS Deployment Runbook (AWS Academy)

Architecture: `S3 (model + data) -> EC2 (Streamlit app) <-> RDS MySQL (predictions)`

Replace every `<PLACEHOLDER>` below with your own values. Never commit real
endpoints, IPs, key files, or passwords back into this repo.

## 1. S3 - Upload model and data

1. Go to **S3** in the AWS Console
2. Click **Create bucket**, name it `<S3_BUCKET>` (for example `heart-disease-ml`), leave defaults, create
3. Upload these files to the bucket:
   - `heart_disease_pipeline.pkl`
   - `heart_disease_dataset.csv`

## 2. RDS MySQL - Create the database

1. Go to **RDS** in the AWS Console
2. Click **Create database**
   - Engine: **MySQL**
   - Template: **Free tier**
   - DB instance identifier: `<RDS_INSTANCE_ID>`
   - Master username: `admin`
   - Master password: choose something strong and save it in your password manager
   - Instance class: `db.t3.micro`
   - Storage: 20 GB (default)
   - Under **Connectivity**: set **Public access** to **Yes**
   - VPC security group: create new or use default
3. Click **Create database** and wait (~5 min for it to become available)
4. Copy the **Endpoint** (looks like `<RDS_INSTANCE_ID>.<ID>.<REGION>.rds.amazonaws.com`)

### Security group setup

The EC2 instance needs to reach RDS on port 3306:

1. Go to the RDS instance's security group
2. Add an **Inbound rule**: Type = MySQL/Aurora, Port = 3306, Source = the EC2 security group (or `0.0.0.0/0` for testing only)

### Create the predictions table

From your local machine (or the EC2 instance after setup):

```bash
mysql -h <RDS_ENDPOINT> -u admin -p < setup_db.sql
```

Or paste the contents of `setup_db.sql` into any MySQL client.

## 3. EC2 - Launch the app server

1. Go to **EC2** in the AWS Console
2. Click **Launch instance**
   - Name: `<EC2_INSTANCE_NAME>` (for example `heart-disease-app`)
   - AMI: **Amazon Linux 2023** (default)
   - Instance type: `t2.micro` (free tier)
   - Key pair: create a new one and download the `.pem` file. Keep it OUTSIDE the repo.
   - Under **Network settings** -> **Security group**, add these inbound rules:
     - SSH (port 22) from your IP
     - Custom TCP (port 8501) from `0.0.0.0/0` (Streamlit)
3. Launch the instance, then copy its **Public IPv4 address** (`<EC2_PUBLIC_IP>`).

### SSH into the instance

```bash
chmod 400 <KEY_FILE>.pem
ssh -i <KEY_FILE>.pem ec2-user@<EC2_PUBLIC_IP>
```

### Install dependencies

```bash
sudo yum update -y
sudo yum install -y python3-pip mysql
pip3 install streamlit scikit-learn pandas pymysql joblib boto3 plotly
```

### Get the model onto EC2

Option A - download from S3 (if AWS CLI is configured with credentials):

```bash
aws s3 cp s3://<S3_BUCKET>/heart_disease_pipeline.pkl .
```

Option B - SCP from your local machine (works in Academy without IAM setup):

```bash
scp -i <KEY_FILE>.pem heart_disease_pipeline.pkl ec2-user@<EC2_PUBLIC_IP>:~
```

### Upload the Streamlit app

```bash
scp -i <KEY_FILE>.pem src/app.py ec2-user@<EC2_PUBLIC_IP>:~
```

### Set environment variables and run

```bash
export DB_HOST=<RDS_ENDPOINT>
export DB_USER=admin
export DB_PASS=<DB_PASSWORD>
export DB_NAME=heart_disease

streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### Access the app

Open a browser and go to: `http://<EC2_PUBLIC_IP>:8501`

## 4. Demo day checklist

- [ ] EC2 instance is running
- [ ] RDS instance is running
- [ ] Streamlit app is accessible at the public IP
- [ ] Submit a test prediction and verify it appears in the "Recent Predictions" table
- [ ] Verify the prediction was written to RDS (optional: connect with MySQL client and `SELECT * FROM predictions;`)

## Troubleshooting

**Streamlit will not load in browser:**
- Check the EC2 security group allows inbound on port 8501
- Make sure you ran with `--server.address 0.0.0.0` (not just localhost)

**Cannot connect to RDS:**
- Check the RDS security group allows inbound on port 3306 from the EC2 security group
- Verify the RDS instance is in "Available" state
- Double-check the endpoint, username, and password

**AWS Academy session expired:**
- Academy sessions typically last 4 hours. Resources (EC2, RDS) persist between sessions, but you need to restart your session to access the console.
- For demo day: start a new Academy session ~15 min before your slot to make sure everything is up.
