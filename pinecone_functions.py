name: Deploy to Modal

on:
  push:
    branches: [ main ]
    
jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
        
    - name: Install Modal
      run: |
        pip install modal
        
    - name: Authenticate with Modal
      env:
        MODAL_TOKEN: ${{ secrets.MODAL_TOKEN }}
      run: |
        echo "$MODAL_TOKEN" | modal token set --stdin
        
    - name: Deploy to Modal
      run: |
        modal deploy pinecone_functions.py
