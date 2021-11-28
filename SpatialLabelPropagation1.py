class SpatialLabelPropagation():
    def __init__(self):
        # Location is represented as a lat/lon geopy Point
        self.user_id_to_location = {}
    
    def get_geometric_median(coordinates):
        """
        Returns the geometric median of the list of locations.
        """
        n = len(coordinates)
        # The geometric median is only defined for n > 3 points, so just return
        # an arbitrary point if we have fewer
        if n == 1:
            return coordinates[0]
        elif n == 2:
            return coordinates[random.randint(0, 1)]          
        median = None # Point type       
        # Loop through all the points, finding the point that minimizes the
        # geodetic distance to all other points.  By construction median will
        # always be assigned to some non-None value by the end of the loop.
        for i in range(0, n):
            p1 = coordinates[i]
            dist_sum = 0
            for j in range(0, n):
                # Skip self-comparison
                if i == j:
                    continue
                p2 = coordinates[j]
                dist = get_distance(p1, p2)
                dist_sum += dist
                # Abort early if we already know this isn't the median
                if dist_sum > min_distance_sum:
                        break
            if dist_sum < min_distance_sum:
                min_distance_sum = dist_sum
                median = p1
        return median

    def train_model(self, setting, dataset):
        """
        Runs spatial label propagation (SLP) on the bi-directional @mention
        network present in the dataset.  The initial locations for SLP are
        set by identifying individuals with at least five GPS-tagged posts
        within 15km of each other.
        """       
        mention_network = dataset.bi_mention_network()
        all_users = set(mention_network.nodes())
        user_to_home_loc = {user: loc for (user, loc) in dataset.user_home_location_iter()}
        user_to_estimated_location = {}
        user_to_estimated_location.update(user_to_home_loc)
        user_to_next_estimated_location = {}                
        for iteration in range(0, num_iterations):
            num_located_at_start = len(user_to_estimated_location)
            num_processed = 0
            for user_id in all_users:
                self.update_user_location(user_id, mention_network, 
                                          user_to_home_loc,
                                          user_to_estimated_location,
                                          user_to_next_estimated_location)
                num_processed += 1
            num_located_at_end = len(user_to_next_estimated_location)
            # Replace all the old location estimates with what we estimated
            # from this iteration
            user_to_estimated_location.update(user_to_next_estimated_location)       
        return user_to_estimated_location            

    def update_user_location(self, user_id, mention_network,
                             user_to_home_loc, user_to_estimated_location,
                             user_to_next_estimated_location):
        """
        Uses the provided social network and estimated user locations to update
        the location of the specified user_id in the
        user_to_next_estimated_location dict.  Users who have a home location
        (defined from GPS data) will always be updated with their home location.
        """
        # Short-circuit if we already know where this user is located
        # so that we always preserve the "hint" going forward
        if user_id in user_to_home_loc:
            user_to_next_estimated_location[user_id] = user_to_home_loc[user_id]
            return
        # For each of the users in the user's ego network, get their estimated
        # location, if any
        locations = []
        for neighbor_id in mention_network.neighbors_iter(user_id):
            if neighbor_id in user_to_estimated_location:
                locations.append(user_to_estimated_location[neighbor_id])
        # If we have at least one location from the neighbors, use the
        # list of locations to infer a location for this individual.
        if len(locations) > 0:
            median = get_geometric_median(locations)
            user_to_next_estimated_location[user_id] = median

    def get_distance(p1, p2):
      """
      Computes the distance between the two latitude-longitude Points using
      Vincenty's Formula
      """
      return distance.distance(p1, p2).kilometers