class Experiment:

    def __init__(self, model, benchmark_name):
        self.model = model
        self.results = {"accuracy": [], "tokens": []}
    
    def run(self, stream, limit=10000):
        # Logic to iterate, track IoU vs tokens, and log results
        pass
