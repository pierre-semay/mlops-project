# mlops-project

# 1. Login
az login

# 2. Create resource group
az group create --name azure-ai --location germanywestcentral

# 3. Create ML workspace
az ml workspace create --name mlops-project --resource-group azure-ai

# 4. Create service principal for GitHub Actions
az ad sp create-for-rbac --name "github-actions" \
  --role contributor \
  --scopes /subscriptions/<their-subscription-id>/resourceGroups/azure-ai \
  --json-auth

# 5. Set up the kubernetes cluster
k3d cluster create mlops-cluster -p "8080:80@loadbalancer" --agents 2

kubectl get nodes

kubectl apply -f database.yaml

kubectl apply -f deployment.yaml

kubectl apply -f nginx.yaml

kubectl get pods
