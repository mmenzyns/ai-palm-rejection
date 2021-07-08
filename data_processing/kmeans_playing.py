
# https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb

#https://www.linkedin.com/pulse/finding-optimal-number-clusters-k-means-through-elbow-asanka-perera

indices = np.fliplr(np.argwhere(image > 10))

X = indices
nclusters_range = range(1, 5)

wcss = []

for i in nclusters_range:
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)


from scipy.spatial import distance

distances = []
for i in range(0,4):
    p1 = Point(initx=1,inity=wcss[0])
    p2 = Point(initx=10,inity=wcss[9])
    p = Point(initx=i+1,inity=wcss[i])
    distances.append(p.distance_to_line(p1,p2))

# plot the 3 clusters
plt.scatter(
    X[y_km == 0, 1], X[y_km == 0, 0],
    s=50, c='lightgreen',
    label='cluster 1'
)
plt.ylim(0, 13)
plt.xlim(0, 20)

plt.scatter(
    X[y_km == 1, 1], X[y_km == 1, 0],
    s=50, c='orange',
    label='cluster 2'
)
plt.ylim(0, 13)
plt.xlim(0, 20)

plt.scatter(
    X[y_km == 2, 1], X[y_km == 2, 0],
    s=50, c='lightblue',
    label='cluster 3'
)
plt.ylim(0, 13)
plt.xlim(0, 20)

plt.scatter(
    X[y_km == 3, 1], X[y_km == 3, 0],
    s=50, c='black',
    label='cluster 3'
)
plt.ylim(0, 13)
plt.xlim(0, 20)


fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(indices.T[0], indices.T[1], z, cmap=cm.coolwarm)