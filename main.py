import numpy as np

class KMedoids:
    def __init__(self, filename, size):
        self.distances = np.loadtxt(filename, delimiter=',')
        self.size = size
    
    def optimize(self):
        max_s = -np.Inf
        max_k = 0
        k = 2
        count = 5
        
        while count > 0:
            print('Trying k=', k, '...\t', sep='', end='')
            self.build(k)
            self.swap()
            curr = self.silhouette_coef()
            print('Silhouette coef:', curr)
            if curr > max_s:
                max_s = curr
                max_k = k
                count = 5
            k += 1
            count -= 1
        return max_k, max_s
    
    def update_closests(self):
        self.closest_dist.fill(np.Inf)
        self.second_closest.fill(np.Inf)
        
        for i in range(self.size):
            for j in range(len(self.medoids)):
                medoid = self.medoids[j]
                if self.distances[i, medoid] < self.closest_dist[i]:
                    self.second_closest[i] = self.closest_dist[i]
                    self.closest_dist[i] = self.distances[i, medoid]
                    self.closest[i] = j
                elif self.distances[i, medoid] < self.second_closest[i]:
                    self.second_closest[i] = self.distances[i, medoid]
    
    def add_medoid(self, point):
        self.medoids.append(point)
        self.update_closests()
    
    def swap_medoid(self, i, h):
        self.medoids[i] = h
        self.update_closests()
    
    def build(self, k):
        self.k = k
        self.medoids = []
        self.closest = np.empty(self.size, dtype=np.int32)
        self.closest_dist = np.full(self.size, np.Inf)
        self.second_closest = np.full(self.size, np.Inf)
        
        # Sum by rows (0)
        sums = np.sum(self.distances, axis=0)
        amin = np.argmin(sums)
        self.add_medoid(amin)
        
        for i in range(self.k - 1):
            tds = np.Inf
            candidate = 0
            for c in range(self.size):
                if c in self.medoids:
                    continue
                
                delta = self.distances[c] - self.closest_dist
                delta = np.delete(delta, c)
                td = np.sum(delta[delta < 0])
                
                if td < tds:
                    tds = td
                    candidate = c
            self.add_medoid(candidate)
    
    def compute_losses(self):
        losses = np.zeros(self.k)
        for i in range(self.size):
            losses[self.closest[i]] += self.second_closest[i] - self.closest_dist[i]
        
        return losses
    
    def swap(self):
        last = None
        first = True
        tds = self.compute_losses()
        
        while True:
            for i in range(self.size):
                if i == last or (last is None and not first):
                    return
                if i in self.medoids:
                    continue
                
                dtd = np.copy(tds)
                
                dists = self.distances[i] - self.closest_dist
                # Return True or false in array for each cell
                fcond = dists < 0 
                # Sum all the True values from fcond
                tdc = np.sum(dists[fcond]) 
                
                # Sub 2 arrays in same size
                ns_diff = self.closest_dist - self.second_closest
                sdist = self.distances[i] - self.second_closest
                
                ns_diff = ns_diff * fcond
                sdist = sdist * ~fcond * (sdist < 0)
                
                total = ns_diff + sdist
                for j in range(self.k):
                    # Sum all value that in same cluster with j (True to j)
                    dtd[j] += np.sum(total[self.closest == j])
                
                argmin = np.argmin(dtd)
                dtd[argmin] += tdc
                # Use a threshold close to zero to avoid problems with
                # rounding errors
                if dtd[argmin] < -0.00001:
                    self.swap_medoid(argmin, i)
                    tds = self.compute_losses()
                    last = i
            first = False
    
    def silhouette_a(self, i):
        # Bool array 
        # True if i and other in same cluster
        # Else False
        cond = self.closest == self.closest[i]
        # Return how much celles is'nt 0
        cluster_size = np.count_nonzero(cond)
        if cluster_size == 1:
            return 0, 1
        # Return sum of the row i of dist matrix
        # if its in the same cluster
        sumv = np.sum(self.distances[i, np.nonzero(cond)])
        return sumv / (cluster_size - 1), cluster_size
    
    def silhouette_b(self, i):
        minv = np.Inf
        for m in range(self.k):
            if m == self.closest[i]:
                continue
            
            # Bool array 
            # True if medoid m is closest
            # Else False
            cond = self.closest == m
            # Return how much celles is'nt 0

            count = np.count_nonzero(cond)
            # Return sum of the row i of dist matrix
            # if its in the same cluster
            sumv = np.sum(self.distances[i, np.nonzero(cond)])
            # mean avg
            mean = sumv / count
            if mean < minv:
                minv = mean
        return minv
    
    def silhouette_s(self, i):
        a, cluster_size = self.silhouette_a(i)
        if cluster_size == 1:
            return 0
        
        b = self.silhouette_b(i)
        
        return (b - a) / max(a, b)
    
    def silhouette_coef(self):
        sumv = 0.0 # Silhouettes sum
        for i in range(self.size):
            sumv += self.silhouette_s(i)
        return sumv / self.size
        

def main():
    print('Loading data... ', end='')
    km = KMedoids('dist_matrix(10000x10000).txt', 10000)
    print('Done')
    print('Finding optimal k')
    k, sil = km.optimize()
    print('Final result: k=', k, ' with silhouette coefficient: ', sil, sep='')

if __name__ == '__main__':
    main()
