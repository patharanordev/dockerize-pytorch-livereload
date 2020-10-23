# gunicorn \
# --bind=0.0.0.0:8000 \
# --workers=4 \
# --reload 'server:app' \
# --name='MLaaS'

gunicorn -b 0.0.0.0:8000 --chdir ./ --reload 'server:app'