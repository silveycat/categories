# Analysis and simulation code for experiments reported in 'Communication increases category structure and alignment
# only when combined with cultural transmission'

# import data and relevant modules
import data as d
import copy
import random
import numpy
from tqdm import tqdm

# dictionary indexing conditions to final category systems
systems_dict = {
    'exp1i': d.exp1_individual_final,
    'exp1c': d.exp1_communication_final,
    'exp2t': d.exp2_transmission_final,
    'exp2c': d.exp2_communication_final
}

# dictionary indexing conditions to full runs
full_runs = {
    'exp1': d.exp1_communication_full,
    'exp2t': d.exp2_transmission_full,
    'exp2c': d.exp2_communication_full
}


# Specificity

# Simply counts number of categories in each system within a condition


def specificity(condition):
    # set up dictionary for collecting results
    results = {}
    # get category systems from the condition of interest
    systems = systems_dict[condition]
    for s in systems:
        # take length of the unique set of category indicators
        spec = len(set(systems[s]))
        # this is the number of categories in the system
        results[s] = spec
    return results


# Convexity

# this analysis code has 3 parts:
# 1) convexity, a function that returns true convexity of a system
# 2) random_convexity, a function that returns the average convexity for random shuffles of a system
# 3) max_convexity, a function that runs a simulation to find the maximum convexity for a system

# first, we need to know which images neighbour each other in the space

neighbours = [
    # image 0
    [1, 5, 6],
    # image 1
    [0, 2, 5, 6, 7],
    # image 2
    [1, 3, 6, 7, 8],
    # image 3
    [2, 4, 7, 8, 9],
    # image 4
    [3, 8, 9],
    # image 5
    [0, 1, 6, 10, 11],
    # image 6
    [0, 1, 2, 5, 7, 10, 11, 12],
    # image 7
    [1, 2, 3, 6, 8, 11, 12, 13],
    # image 8
    [2, 3, 4, 7, 9, 12, 13, 14],
    # image 9
    [3, 4, 8, 13, 14],
    # image 10
    [5, 6, 11, 15, 16],
    # image 11
    [5, 6, 7, 10, 12, 15, 16, 17],
    # image 12
    [6, 7, 8, 11, 13, 16, 17, 18],
    # image 13
    [7, 8, 9, 12, 14, 17, 18, 19],
    # image 14
    [8, 9, 13, 18, 19],
    # image 15
    [10, 11, 16, 20, 21],
    # image 16
    [10, 11, 12, 15, 17, 20, 21, 22],
    # image 17
    [11, 12, 13, 16, 18, 21, 22, 23],
    # image 18
    [12, 13, 14, 17, 19, 22, 23, 24],
    # image 19
    [13, 14, 18, 23, 24],
    # image 20
    [15, 16, 21],
    # image 21
    [15, 16, 17, 20, 22],
    # image 22
    [16, 17, 18, 21, 23],
    # image 23
    [17, 18, 19, 22, 24],
    # image 24
    [18, 19, 23]]


def convexity(system):
    # get the set of category identifiers used in this system
    category_identifiers = set(system)
    # list for collecting lists containing members of each category
    master_cat_members = []
    # also keep count of how many 1-image categories there are
    single_member_categories = 0
    # go through each category identifier
    for cat in category_identifiers:
        # set up list for collecting members of this category
        category_members = []
        for i in range(len(system)):
            if system[i] == cat:
                category_members.append(i)
        # add this list to the master list
        if len(category_members) > 1:
            master_cat_members.append(category_members)
        # unless there is only one member of this category
        else:
            single_member_categories += 1
    # for each category member, what proportion of its neighbours are in the category?
    neighbour_proportions = []
    for cat in master_cat_members:
        proportions = []
        for member in cat:
            # initialise neighbour count at 0
            count = 0
            # check through neighbours to find which are in the same category
            for neighbour in neighbours[member]:
                if neighbour in cat:
                    count += 1
            # calculate proportion
            prop = count / float(len(neighbours[member]))
            proportions.append(prop)
        # average these proportions over all members of this category
        category_average_proportion = sum(proportions) / float(len(proportions))
        neighbour_proportions.append(category_average_proportion)
    # every 1-member cat has 0 neighbours in the category
    for x in range(single_member_categories):
        neighbour_proportions.append(0)
    # return the average over all categories
    return sum(neighbour_proportions) / float(len(neighbour_proportions))


# random convexity function - analyses use 100,000 rounds
def random_convexity(system, rounds):
    # get true convexity
    true_convexity = convexity(system)
    # copy this system
    copycat = copy.copy(system)
    # create list to collect distribution of convexity measures
    distribution = []
    tqdm.write('Finding expected convexity')
    for _ in tqdm(range(rounds)):
        # shuffle the copy and take its convexity, append to distribution
        random.shuffle(copycat)
        new_cat = copycat
        distribution.append(convexity(new_cat))
    # return the true convexity and the mean of the distribution
    return true_convexity, sum(distribution) / float(rounds)


# Genetic algorithm to find maximum convexity

# parameters
pop_size = 1000
mutationRate = 2
generations = 2500


# population creator - initially a set number of shuffled copies of true system
def pop_create(system):
    population = []
    for n in range(pop_size):
        new_sys = copy.copy(system)
        random.shuffle(new_sys)
        population.append(new_sys)
    return population


# mutation function - randomly swaps a set number of image category assignments
def mutate(system, rate):
    # first create a clone
    clone = copy.copy(system)
    sequence = list(range(25))
    for m in range(rate):
        # pick 2 random images to swap their categories
        first = random.choice(sequence)
        # remove from sequence so we don't try and swap it with itself
        sequence.remove(first)
        second = random.choice(sequence)
        sequence.remove(second)
        # locate the current category assignments of these images
        first_val = clone[first]
        second_val = clone[second]
        # swap the categories
        clone[first] = second_val
        clone[second] = first_val
    # return the mutated category system
    return clone


# generation turnover function
def generation(population):
    # list for collecting scores
    indexed_scores = []
    # get convexity score for each system in the population
    for system in range(len(population)):
        score = convexity(population[system])
        # add to the list along with the system
        indexed_scores.append([system, score])
    # sort systems by score
    indexed_scores.sort(key=lambda y: y[1])
    max_index = indexed_scores[-1][0]
    max_score = indexed_scores[-1][1]
    best_half = []
    # if more than one system gets best score, pick the first one
    best_system = population[max_index]
    # get second (best) half of systems
    for x in range(int(len(population)/2), len(population)):
        best_half.append(population[indexed_scores[x][0]])
    # the other half of the new population consists of 'mutated' versions of these systems
    new_systems = []
    for system in best_half:
        new_system = mutate(system, mutationRate)
        new_systems.append(new_system)
    best_half.extend(new_systems)
    return best_half, best_system, max_score


# main simulation
# with current parameters this takes around 4-5 minutes per system
# it is generally stable over runs (+/- 0.01)
# this function can also be adapted to inspect the best system to verify it is convex when laid out visually
# to do this, return bests[-1]

def max_convexity(system):
    # set up initial population
    population = pop_create(system)
    # lists for collecting top scores and best systems
    top_scores = []
    bests = []
    # generation loop
    tqdm.write('Finding maximum convexity')
    for _ in tqdm(range(generations)):
        data = generation(population)
        population = data[0]
        top_scores.append(data[2])
        bests.append(data[1])
    return max(top_scores)


# Alignment (also used as measure of category system learnability in Experiment 2)
# Calculates Adjusted Rand index (details in Hubert & Arabie 1985)
# Adjusted Rand index is stochastically bounded by 0 and 1
# when index = 1, clusterings are identical
# when index = 0, clusterings are as similar as you would expect by chance
# argument 'pair' = a list containing the two category systems to be compared

def alignment(pair):
    # number of images
    num_images = len(pair[0])
    # set first category system = system1, second system = system2 (order doesn't affect result)
    system1 = pair[0]
    system2 = pair[1]
    # get the number indicators for categories in each system
    sys1_cat_nums = set(system1)
    sys2_cat_nums = set(system2)
    # find number of items in each category
    # we need this to calculate the expected index
    sys1_cat_counts = []
    sys2_cat_counts = []
    for ind in sys1_cat_nums:
        counts = 0
        for i in system1:
            if i == ind:
                counts += 1
        sys1_cat_counts.append(counts)
    for ind in sys2_cat_nums:
        counts = 0
        for j in system2:
            if j == ind:
                counts += 1
        sys2_cat_counts.append(counts)
    # so this produces a list for each system with the number of items in each category
    # calculate Rand index
    agree_score = 0
    diff_score = 0
    score_list = []
    # build matrix of scores for pairs of images
    for x in range(num_images):
        scores = []
        for y in range(num_images):
            # all scores initially 0
            scores.append(0)
        score_list.append(scores)
    # populate score_list
    # for each possible pair of images
    for i in range(num_images):
        for k in range(num_images):
            # for each of the two systems being compared
            for j in range(len(pair)):
                # if we are trying to compare an image to itself, mark score as 3 (meaningless)
                if i == k:
                    score_list[i][k] = 3
                # if the category indicators for the two images match
                elif pair[j][i] == pair[j][k]:
                    # we add 1 to the agreement index in the score list
                    # this means there are 3 possible values in score list:
                    # 0 = both systems placed these images in different categories
                    # 1 = one system placed these images in the same category, one system in different
                    # 2 = both systems placed these images in the same category
                    score_list[i][k] += 1
    # go through score list
    for y in range(len(score_list)):
        for z in range(len(score_list[y])):
            # only check unique pairs of images
            if z < y:
                continue
            # if score for pair is either 2 or 0, class this as an agreement
            elif score_list[y][z] == len(pair) or score_list[y][z] == 0:
                agree_score += 1
            # otherwise, score will be 1, & this counts as a disagreement
            elif score_list[y][z] < 3:
                diff_score += 1
    # calculate total number of image pairs
    num_pairs = agree_score + diff_score
    # Unadjusted Rand index = number of agreements / total number of pairs
    ri = agree_score/float(num_pairs)
    # calculate expected Rand index
    # formula from Hubert & Arabie (1985), 'Comparing partitions'
    sys1_totals = 0
    sys2_totals = 0
    for s in sys1_cat_counts:
        value = s*(s-1)/2
        sys1_totals += value
    for u in sys2_cat_counts:
        value = u*(u-1)/2
        sys2_totals += value
    expected_ri = 1 + 2 * sys1_totals * sys2_totals / float(pow(num_pairs, 2)) - \
        (sys1_totals + sys2_totals) / float(num_pairs)
    # adjusted index = (actual index - expected index)/(max value of index - expected index)
    try:
        adj_ri = (ri - expected_ri) / float(1 - expected_ri)
    # if the expected ri is 1, the adjusted ri is 0
    except ZeroDivisionError:
        print(pair)
        adj_ri = 0
    return adj_ri


# Convergence = alignment across a whole condition/generation
# within a condition/generation, shuffles pairs at random, finds mean alignment
# repeats this a number of times (100,000 in analysis) to get a distribution of means
# returns mean convergence and confidence intervals
# This takes approx 15 minutes per condition/generation

def convergence(data, rounds):
    # Takes a dictionary as data - extract systems from this and put in a list
    sys_list = []
    for lang in data:
        # print('including %s' % lang)
        sys_list.append(data[lang])
    pairs = list(range(len(sys_list)))
    # set up list to collect means
    mean_dist = []
    for _ in tqdm(range(rounds)):
        alignments = []
        # shuffle pair indices
        random.shuffle(pairs)
        # go through this shuffled list of indices in order
        for pair in range(0, len(pairs), 2):
            pairing = [sys_list[pairs[pair]], sys_list[pairs[pair + 1]]]
            a = alignment(pairing)
            alignments.append(a)
        mean = sum(alignments)/float(len(alignments))
        mean_dist.append(mean)
    # now we have a distribution of means
    # get the standard deviation of this distribution
    grand_mean = sum(mean_dist)/float(len(mean_dist))
    se = numpy.std(mean_dist)
    low_lim = grand_mean - (se * 1.96)
    up_lim = grand_mean + (se * 1.96)
    return grand_mean, low_lim, up_lim


# communicative success

def communicative_success(data):
    # figure out which experiment we're analysing
    if len(data.keys()) > 20:
        # this is experiment 2
        results = {}
        for gen in range(5):
            # dictionary for collecting scores by round
            by_round = {1: [], 2: [], 3: [], 4: []}
            # list for collecting scores in last 2 rounds
            # here, we take the average
            successes_uncorrected = []
            successes_corrected = []
            for chain in range(8):
                p_str = "c" + str(chain + 1) + "g" + str(gen + 1) + "0"
                p_str_2 = "c" + str(chain + 1) + "g" + str(gen + 1) + "1"
                p_dat = data[p_str]['test']
                last_2_p = 0
                for rnd in range(4):
                    round_score = 0
                    for tr in p_dat:
                        if tr[0] == rnd:
                            round_score += tr[4]
                            if rnd > 1:
                                last_2_p += tr[4]
                    by_round[rnd + 1].append(round_score)
                success_uncorrected = last_2_p / float(2)
                successes_uncorrected.append(success_uncorrected)
                # create version corrected for average number of categories
                system1 = systems_dict['exp2c'][p_str]
                system2 = systems_dict['exp2c'][p_str_2]
                average_cats = (len(set(system1)) + len(set(system2))) / float(2)
                success_corrected = success_uncorrected / float(average_cats)
                successes_corrected.append(success_corrected)
            results[gen + 1] = [by_round, successes_uncorrected, successes_corrected]
        return results
    else:
        # this is experiment 1
        # dictionary for collecting scores by round
        by_round = {1: [], 2: [], 3: [], 4: []}
        # list for collecting scores in last 2 rounds
        last_2 = []
        for part in range(1, 20, 2):
            p_str = "p" + str(part) + "p" + str(part + 1)
            p_dat = data[p_str]
            last_2_p = 0
            for rnd in range(4):
                round_score = 0
                for tr in p_dat:
                    if tr[0] == rnd:
                        round_score += tr[4]
                        if rnd > 1:
                            last_2_p += tr[4]
                by_round[rnd + 1].append(round_score)
            last_2.append(last_2_p)
        return by_round, last_2


# similarity-based scoring function from experiment
similarities = [[15, 14, 12, 9, 6, 14, 13, 11, 8, 5, 12, 11, 10, 7, 3, 9, 8, 7, 4, 2, 6, 5, 3, 2, 1],
                [14, 15, 14, 12, 9, 13, 14, 13, 11, 8, 11, 12, 11, 10, 7, 8, 9, 8, 7, 4, 5, 6, 5, 3, 2],
                [12, 14, 15, 14, 12, 11, 13, 14, 13, 11, 10, 11, 12, 11, 10, 7, 8, 9, 8, 7, 3, 5, 6, 5, 3],
                [9, 12, 14, 15, 14, 8, 11, 13, 14, 13, 7, 10, 11, 12, 11, 4, 7, 8, 9, 8, 2, 3, 5, 6, 5],
                [6, 9, 12, 14, 15, 5, 8, 11, 13, 14, 3, 7, 10, 11, 12, 2, 4, 7, 8, 9, 1, 2, 3, 5, 6],
                [14, 13, 11, 8, 5, 15, 14, 12, 9, 6, 14, 13, 11, 8, 5, 12, 11, 10, 7, 3, 9, 8, 7, 4, 2],
                [13, 14, 13, 11, 8, 14, 15, 14, 12, 9, 13, 14, 13, 11, 8, 11, 12, 11, 10, 7, 8, 9, 8, 7, 4],
                [11, 13, 14, 13, 11, 12, 14, 15, 14, 12, 11, 13, 14, 13, 11, 10, 11, 12, 11, 10, 7, 8, 9, 8, 7],
                [8, 11, 13, 14, 13, 9, 12, 14, 15, 14, 8, 11, 13, 14, 13, 7, 10, 11, 12, 11, 4, 7, 8, 9, 8],
                [5, 8, 11, 13, 14, 6, 9, 12, 14, 15, 5, 8, 11, 13, 14, 3, 7, 10, 11, 12, 2, 4, 7, 8, 9],
                [12, 11, 10, 7, 3, 14, 13, 11, 8, 5, 15, 14, 12, 9, 6, 14, 13, 11, 8, 5, 12, 11, 10, 7, 3],
                [11, 12, 11, 10, 7, 13, 14, 13, 11, 8, 14, 15, 14, 12, 9, 13, 14, 13, 11, 8, 11, 12, 11, 10, 7],
                [10, 11, 12, 11, 10, 11, 13, 14, 13, 11, 12, 14, 15, 14, 12, 11, 13, 14, 13, 11, 10, 11, 12, 11, 10],
                [7, 10, 11, 12, 11, 8, 11, 13, 14, 13, 9, 12, 14, 15, 14, 8, 11, 13, 14, 13, 7, 10, 11, 12, 11],
                [3, 7, 10, 11, 12, 5, 8, 11, 13, 14, 6, 9, 12, 14, 15, 5, 8, 11, 13, 14, 3, 7, 10, 11, 12],
                [9, 8, 7, 4, 2, 12, 11, 10, 7, 3, 14, 13, 11, 8, 5, 15, 14, 12, 9, 6, 14, 13, 11, 8, 5],
                [8, 9, 8, 7, 4, 11, 12, 11, 10, 7, 13, 14, 13, 11, 8, 14, 15, 14, 12, 9, 13, 14, 13, 11, 8],
                [7, 8, 9, 8, 7, 10, 11, 12, 11, 10, 11, 13, 14, 13, 11, 12, 14, 15, 14, 12, 11, 13, 14, 13, 11],
                [4, 7, 8, 9, 8, 7, 10, 11, 12, 11, 8, 11, 13, 14, 13, 9, 12, 14, 15, 14, 8, 11, 13, 14, 13],
                [2, 4, 7, 8, 9, 3, 7, 10, 11, 12, 5, 8, 11, 13, 14, 6, 9, 12, 14, 15, 5, 8, 11, 13, 14],
                [6, 5, 3, 2, 1, 9, 8, 7, 4, 2, 12, 11, 10, 7, 3, 14, 13, 11, 8, 5, 15, 14, 12, 9, 6],
                [5, 6, 5, 3, 2, 8, 9, 8, 7, 4, 11, 12, 11, 10, 7, 13, 14, 13, 11, 8, 14, 15, 14, 12, 9],
                [3, 5, 6, 5, 3, 7, 8, 9, 8, 7, 10, 11, 12, 11, 10, 11, 13, 14, 13, 11, 12, 14, 15, 14, 12],
                [2, 3, 5, 6, 5, 4, 7, 8, 9, 8, 7, 10, 11, 12, 11, 8, 11, 13, 14, 13, 9, 12, 14, 15, 14],
                [1, 2, 3, 5, 6, 2, 4, 7, 8, 9, 3, 7, 10, 11, 12, 5, 8, 11, 13, 14, 6, 9, 12, 14, 15]]


# communicative success simulation: for getting success scores for Transmission Alone participants

# first, we need to determine the prototype for each category
# i.e. the image that is maximally similar to all images in the category
# if more than one possible prototype, picks the first occurring in list of candidates

def prototype(agent, category):
    # set up list for collecting candidate images
    candidates = []
    # set up list for collecting average similarities of category members to all other members
    average_sims = []
    # collect all images that belong to the category in question
    for item in range(len(agent)):
        if agent[item] == category:
            candidates.append(item)
    # if the category only has one member, this is the prototype by default
    if len(candidates) == 1:
        return candidates[0]
    else:
        for member in candidates:
            member_sims = []
            # check distances between this category member and other category members
            for comparison in candidates:
                if member != comparison:
                    member_sims.append(similarities[member][comparison])
            average_sims.append(sum(member_sims) / float(len(member_sims)))
        return candidates[average_sims.index(max(average_sims))]


# function to make a dictionary of prototypes for each word
def proto_dict(agent):
    proto = {}
    # go through all images in system
    for item in agent:
        # check if this image's category already has a prototype
        if item not in proto:
            # if not, use the prototype function to get it & store it in dict
            proto[item] = prototype(agent, item)
    return proto


# is_even function (for setting up balanced communication lists)
def is_even(number):
    if number % 2 == 0:
        return True
    else:
        return False


# success simulation
def success_sim(pair):
    agent_1 = pair[0]
    agent_2 = pair[1]
    # get prototypes for each agent
    pair1dict = proto_dict(agent_1)
    pair2dict = proto_dict(agent_2)
    comm_lists = []
    total_score = 0
    successes = []
    # initialise one agent as sender and the other as receiver
    sender = agent_1
    receiver = agent_2
    # use the current receiver's prototype dictionary
    dictionary = pair2dict
    # set up counterbalanced orders for communication, as in experiment
    for iteration in range(2):
        comm_list = list(range(25))
        random.shuffle(comm_list)
        comm_lists.append(comm_list)
        a_list = []
        b_list = []
        back_list = list(range(25))
        for index in range(len(comm_list)):
            if is_even(index):
                a_list.append(comm_list[index])
            else:
                b_list.append(comm_list[index])
        random.shuffle(a_list)
        random.shuffle(b_list)
        k = 0
        m = 1
        for e in range(len(a_list)):
            back_list[k] = a_list[e]
            k += 2
        for q in range(len(b_list)):
            back_list[m] = b_list[q]
            m += 2
        comm_lists.append(back_list)
    # main communication function
    round_totals = []
    for comm_list in comm_lists:
        round_total = 0
        for target in comm_list:
            # sender finds word for the target based on their categories
            word = sender[target]
            # if the receiver doesn't have this word, they make a random pick
            if word not in receiver:
                pick = random.choice(comm_list)
            else:
                # receiver picks similarity-central category member (prototype)
                pick = dictionary[word]
            # sender and receiver swap roles
            if sender == agent_1:
                sender = agent_2
                receiver = agent_1
                dictionary = pair1dict
            elif sender == agent_2:
                sender = agent_1
                receiver = agent_2
                dictionary = pair2dict
            # get score from the similarities matrix
            score = similarities[target][pick]
            # add this to cumulative success score
            total_score += score
            round_total += score
            successes.append(total_score)
        round_totals.append(round_total)
    # correct score for average number of categories of the pair
    success_uncorrected = sum(round_totals) / float(len(round_totals))
    average_cats = (len(set(agent_1)) + len(set(agent_2))) / float(2)
    success_corrected = success_uncorrected / average_cats
    return success_uncorrected, success_corrected


# learnability
def learnability(data):
    results = {}
    # learnability is undefined for generation 1
    for gen in range(2, 6):
        # list for collecting similarities between trained & output category systems
        sims = []
        for chain in range(1, 9):
            for participant in range(2):
                sys_key = "c" + str(chain) + "g" + str(gen) + str(participant)
                # trained system is from same chain, generation before, with 1 suffix
                trained_sys_key = "c" + str(chain) + "g" + str(gen - 1) + str(1)
                # calculate similarity = alignment between trained & output system
                similarity = alignment([data[sys_key], data[trained_sys_key]])
                sims.append(similarity)
        results[gen] = sims
    return results


# training scores - for checking for population differences
def training_scores():
    train_file = open("output/experiment_2_training_scores.txt", "w")
    train_file.write("Condition\tNumCats\tScore\n")
    for condition in ["t", "c"]:
        for chain in range(1, 9):
            for gen in range(1, 6):
                for participant in range(2):
                    p_key = "c" + str(chain) + "g" + str(gen) + str(participant)
                    p_run = full_runs['exp2%s' % condition][p_key]
                    # find number of categories in training language for this participant
                    if gen == 1:
                        train_lang_cats = 25
                    else:
                        train_lang_key = "c" + str(chain) + "g" + str(gen - 1) + "1"
                        train_lang_cats = len(set(systems_dict['exp2%s' % condition][train_lang_key]))
                    # isolate training trials
                    train_trials = p_run['train']
                    train_score = 0
                    for t in train_trials:
                        # add 1 or 0 depending on success
                        train_score += t[4]
                    train_file.write("%s\t%i\t%i\n" % (condition, train_lang_cats, train_score))
    train_file.close()


if __name__ == '__main__':

    # run each analysis and output the results to files

    print('Specificity: Analysing experiment 1')

    specificity_exp1 = open("output/experiment_1_specificity.txt", "w")

    specificity_exp1.write("Condition\tSpecificity\n")

    specificity_exp1_results = {
        'i': specificity("exp1i"),
        'c': specificity("exp1c")
    }

    for c in ["i", "c"]:
        for p in range(20):
            key = 'p' + str(p + 1)
            result = specificity_exp1_results[c][key]
            specificity_exp1.write("%s\t%i\n" % (c, result))
    specificity_exp1.close()

    # run this analysis for experiment 2 and output the results to a file

    print('Specificity: Analysing experiment 2')

    specificity_exp2 = open("output/experiment_2_specificity.txt", "w")

    specificity_exp2.write("Condition\tGeneration\tSpecificity\n")

    specificity_exp2_results = {
        't': specificity("exp2t"),
        'c': specificity("exp2c")
    }

    for c in ["t", "c"]:
        for ch in range(8):
            for g in range(5):
                for p in range(2):
                    key = "c" + str(ch + 1) + "g" + str(g + 1) + str(p)
                    result = specificity_exp2_results[c][key]
                    specificity_exp2.write("%s\t%i\t%i\n" % (c, g + 1, result))
    specificity_exp2.close()

    # Convexity

    # experiment 1 convexity code takes around 3-4 minutes per participant, 2-3 hours total
    convexity_exp1 = open("output/experiment_1_convexity.txt", "w")

    convexity_exp1.write("Condition\tTrueConvexity\tRandomConvexity\tMaxConvexity\tCorrected\n")

    for cond in ["i", "c"]:
        for p in range(20):
            p_string = "p" + str(p + 1)
            print('\nConvexity: Analysing experiment 1 condition %s participant %s' % (cond, p_string))
            current_system = systems_dict['exp1%s' % cond][p_string]
            true_c = convexity(current_system)
            random_c = random_convexity(current_system, 100000)[1]
            max_c = max_convexity(current_system)
            corrected = (true_c - random_c)/(max_c - random_c)
            convexity_exp1.write('%s\t%f\t%f\t%f\t%f\n' % (cond, true_c, random_c, max_c, corrected))
    convexity_exp1.close()

    # experiment 2 convexity code takes around 3-4 minutes per participant, 8-11 hours total

    convexity_exp2 = open("output/experiment_2_convexity.txt", "w")

    convexity_exp2.write("Condition\tGeneration\tTrueConvexity\tRandomConvexity\tMaxConvexity\tCorrected\n")

    for cond in ["t", "c"]:
        for c in range(8):
            for g in range(5):
                for p in range(2):
                    p_string = "c" + str(c + 1) + "g" + str(g + 1) + str(p)
                    print('\nConvexity: Analysing experiment 2 participant %s' % p_string)
                    current_system = systems_dict['exp2%s' % cond][p_string]
                    true_c = convexity(current_system)
                    random_c = random_convexity(current_system, 100000)[1]
                    max_c = max_convexity(current_system)
                    corrected = (true_c - random_c)/(max_c - random_c)
                    convexity_exp2.write('%s\t%i\t%f\t%f\t%f\t%f\n' %
                                         (cond, g + 1, true_c, random_c, max_c, corrected))
    convexity_exp2.close()

    print('Alignment: Analysing experiment 1')

    alignment_exp1 = open("output/experiment_1_alignment.txt", "w")
    alignment_exp1.write("Condition\tAdjustedRandIndex\n")

    for cond in ["i", "c"]:
        for p in range(0, 20, 2):
            p_string_1 = "p" + str(p + 1)
            p_string_2 = "p" + str(p + 2)
            system_1 = systems_dict['exp1%s' % cond][p_string_1]
            system_2 = systems_dict['exp1%s' % cond][p_string_2]
            align = alignment([system_1, system_2])
            alignment_exp1.write('%s\t%f\n' % (cond, align))
    alignment_exp1.close()

    print('Alignment: analysing experiment 2')

    alignment_exp2 = open("output/experiment_2_alignment.txt", "w")
    alignment_exp2.write("Condition\tGeneration\tAdjustedRandIndex\n")

    for cond in ["t", "c"]:
        for c in range(1, 9):
            for g in range(1, 6):
                p_string_1 = "c" + str(c) + "g" + str(g) + str(0)
                p_string_2 = "c" + str(c) + "g" + str(g) + str(1)
                system_1 = systems_dict['exp2%s' % cond][p_string_1]
                system_2 = systems_dict['exp2%s' % cond][p_string_2]
                align = alignment([system_1, system_2])
                alignment_exp2.write('%s\t%i\t%f\n' % (cond, g, align))
    alignment_exp2.close()

    # exp1 convergence code takes approx 15 minutes per condition, 30 minutes total
    converge_exp1 = open("output/experiment_1_convergence.txt", "w")
    converge_exp1.write("Condition\tConvergence\tLowLim\tUpLim\n")

    conditions = ["i", "c"]

    for cond in range(len(conditions)):
        print("Convergence: Analysing experiment 1 condition %s" % conditions[cond])
        cat_systems = systems_dict['exp1%s' % conditions[cond]]
        converge = convergence(cat_systems, 100000)
        converge_exp1.write("%s\t%f\t%f\t%f\n" % (conditions[cond], converge[0], converge[1], converge[2]))
    converge_exp1.close()

    # exp2 convergence code takes approx 15 minutes per generation/condition, 2-3 hours total
    converge_exp2 = open("output/experiment_2_convergence.txt", "w")
    converge_exp2.write("Condition\tGeneration\tConvergence\tLowLim\tUpLim\n")

    for cond in ["t", "c"]:
        for g in range(1, 6):
            print("Convergence: Analysing experiment 2 condition %s generation %i" % (cond, g))
            gen_dict = {}
            for c in range(1, 9):
                for p in range(2):
                    p_string = "c" + str(c) + "g" + str(g) + str(p)
                    gen_dict[p_string] = systems_dict['exp2%s' % cond][p_string]
            converge = convergence(gen_dict, 100000)
            converge_exp2.write("%s\t%i\t%f\t%f\t%f\n" % (cond, g, converge[0], converge[1], converge[2]))
    converge_exp2.close()

    print('Communicative success: Analysing experiment 1')

    success_exp1 = open("output/experiment_1_success_by_round.txt", "w")
    success_exp1.write("Round\tScore\n")
    success = communicative_success(full_runs['exp1'])
    for r in range(1, 5):
        for p in success[0][r]:
            success_exp1.write('%i\t%i\n' % (r, p))
    success_exp1.close()

    success_2_exp1 = open("output/experiment_1_success_last_2.txt", "w")
    success_2_exp1.write("Success\n")
    for p in success[1]:
        success_2_exp1.write('%i\n' % p)
    success_2_exp1.close()

    print('Communicative success: Analysing experiment 2')

    success_exp2 = open("output/experiment_2_success_by_round.txt", "w")
    success_exp2.write("Generation\tRound\tScore\n")
    success_output = communicative_success(full_runs['exp2c'])
    for g in range(1, 6):
        success_by_round = success_output[g][0]
        for r in range(1, 5):
            for p in success_by_round[r]:
                success_exp2.write("%i\t%i\t%i\n" % (g, r, p))
    success_exp2.close()

    success_2_exp2 = open("output/experiment_2_success_last_2.txt", "w")
    success_2_exp2.write("Generation\tSuccessRaw\tSuccessCorr\n")
    success_output = communicative_success(full_runs['exp2c'])
    for g in range(1, 6):
        success = success_output[g]
        success_raw = success[1]
        success_corr = success[2]
        for p in range(len(success_raw)):
            success_2_exp2.write("%i\t%f\t%f\n" % (g, success_raw[p], success_corr[p]))
    success_2_exp2.close()

    sim_success = open("output/experiment_2_sim_success.txt", "w")
    sim_success.write("Generation\tSimSuccessRaw\tSimSuccessCorr\n")
    for g in range(1, 6):
        for c in range(1, 9):
            pair1key = 'c' + str(c) + 'g' + str(g) + '0'
            pair2key = 'c' + str(c) + 'g' + str(g) + '1'
            success = success_sim([systems_dict['exp2t'][pair1key], systems_dict['exp2t'][pair2key]])
            sim_success.write("%i\t%f\t%f\n" % (g, success[0], success[1]))
    sim_success.close()

    learnability_results = open("output/experiment_2_learnability.txt", "w")
    learnability_results.write("Condition\tGeneration\tLearnability\n")
    for exp in ['t', 'c']:
        learnability_output = learnability(systems_dict['exp2%s' % exp])
        # learnability is undefined for generation 1
        for g in range(2, 6):
            outputs = learnability_output[g]
            for o in outputs:
                learnability_results.write("%s\t%i\t%f\n" % (exp, g, o))
    learnability_results.close()

    print('Training scores: analysing experiment 2')

    training_scores()
