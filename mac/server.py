# Machine Learning as a Service (MLaaS)
# Note:
# 
# In this scope is in gunicorn live reloading service.
# So any files which are modified from outside container
# or host machine, it will be re-compiled within container too.

import falcon
import tests
import ml

# Falcon follows the REST architectural style, meaning (among
# other things) that you think in terms of resources and state
# transitions, which map to HTTP verbs.
class MLaaS(object):
    def on_get(self, req, resp):
        """Handles GET requests"""
        resp.status = falcon.HTTP_200 # This is the default status
        resp.body = ('\nMLaaS available...\n')
            
# falcon.API instances are callable WSGI apps
app = falcon.API()
# Resources are represented by long-lived class instances
things = MLaaS()
# things will handle all requests to the '/health' URL path
app.add_route('/health', things)