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