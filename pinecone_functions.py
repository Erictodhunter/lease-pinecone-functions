name: Deploy to Modal

on:
  push:
    branches: [ main ]
    
jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to Modal
      uses: modal-labs/modal-action@v1
      with:
        modal-token-id: ${{ secrets.MODAL_TOKEN_ID }}
        modal-token-secret: ${{ secrets.MODAL_TOKEN_SECRET }}
        command: modal deploy pinecone_functions.py
