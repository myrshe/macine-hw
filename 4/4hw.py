#
# Используя генетические или муравьиные алгоритмы найти маршрут проходящий через точки.
# Сами точки берутся из файла/кода/...
# У точек есть вес (приоритет посещения).
# При этом маршрут должен проходится пешеходом/автомобилистом/на транспорте определенное заданное количество времени.
# Необходимо найти путь с наивысшим приоритетом с заданным ограничением времени.
# Маршрут должен быть построен на карте (использовать api).

import requests
import folium
import random
import numpy as np
from typing import List, Dict, Tuple
from itertools import combinations
from deap import base, creator, tools, algorithms


class OSRMClient:

    def __init__(self):
        self.base_url = "http://router.project-osrm.org/route/v1/driving/"
        self.profiles = {
            'car': 'driving',
            'bike': 'cycling',
            'foot': 'walking'
        }

    def get_route_info(self, points: List[Tuple[float, float]], profile: str = 'car') -> Dict:
        if profile not in self.profiles:
            raise ValueError(f"Invalid profile. Available: {list(self.profiles.keys())}")

        coords_str = ";".join([f"{lon},{lat}" for lat, lon in points])
        url = f"{self.base_url}{coords_str}?overview=full"

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            if data['code'] != 'Ok':
                raise ValueError(f"OSRM error: {data.get('message', 'Unknown error')}")

            route = data['routes'][0]
            return {
                'distance': route['distance'] / 1000,
                'duration': route['duration'] / 60,
                'geometry': route['geometry']
            }
        except Exception as e:
            raise ValueError(f"OSRM request failed: {str(e)}")


class GeneticOptimizer:

    def __init__(self, points: List[Dict], max_time: float, profile: str = 'car'):
        self.points = points
        self.max_time = max_time
        self.profile = profile
        self.osrm = OSRMClient()
        self.distance_cache = {}
        self.time_cache = {}

        self._cache_distances()

        self._setup_genetic_algorithm()

    def _cache_distances(self):
        """Кэширование расстояний и времен между всеми парами точек"""
        for i, j in combinations(range(len(self.points)), 2):
            point1 = self.points[i]
            point2 = self.points[j]
            key = frozenset({i, j})

            try:
                route_info = self.osrm.get_route_info(
                    [(point1['lat'], point1['lon']), (point2['lat'], point2['lon'])],
                    self.profile
                )
                self.distance_cache[key] = route_info['distance']
                self.time_cache[key] = route_info['duration']
            except Exception as e:
                print(f"Error getting route between {i} and {j}: {str(e)}")
                self.distance_cache[key] = float('inf')
                self.time_cache[key] = float('inf')

    def get_distance(self, i: int, j: int) -> float:
        if i == j:
            return 0
        key = frozenset({i, j})
        return self.distance_cache.get(key, float('inf'))

    def get_time(self, i: int, j: int) -> float:
        if i == j:
            return 0
        key = frozenset({i, j})
        return self.time_cache.get(key, float('inf'))

    def _setup_genetic_algorithm(self):

        creator.create("FitnessMax", base.Fitness, weights=(1.0, -0.5))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        def init_individual():
            ind = sorted(range(len(self.points)),
                         key=lambda x: -self.points[x]['priority'])
            # Добавляем случайные перестановки
            for _ in range(random.randint(1, 3)):
                i, j = random.sample(range(len(ind)), 2)
                ind[i], ind[j] = ind[j], ind[i]
            return ind

        toolbox.register("individual", tools.initIterate, creator.Individual, init_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("mate", self._custom_crossover)
        toolbox.register("mutate", self._custom_mutation)
        toolbox.register("select", tools.selTournament, tournsize=5)
        toolbox.register("evaluate", self._evaluate_individual)

        self.toolbox = toolbox

    def _custom_crossover(self, ind1, ind2):
        size = len(ind1)
        if size != len(ind2):
            return ind1, ind2

        # Выбираем случайный сегмент из первого родителя
        a, b = sorted(random.sample(range(size), 2))
        segment = ind1[a:b]

        # Создаем ребенка
        child = [None] * size
        child[a:b] = segment

        # Заполняем оставшиеся позиции из второго родителя
        remaining = [x for x in ind2 if x not in segment]
        ptr = 0

        for i in range(size):
            if child[i] is None:
                if ptr < len(remaining):
                    child[i] = remaining[ptr]
                    ptr += 1
                else:
                    # Если остались незаполненные позиции
                    child[i] = [x for x in ind2 if x not in child[:i]][0]

        return creator.Individual(child), creator.Individual(ind2)

    def _custom_mutation(self, individual):
        size = len(individual)

        if random.random() < 0.5:
            i, j = random.sample(range(size), 2)
            individual[i], individual[j] = individual[j], individual[i]
        else:
            for _ in range(3):
                for k in range(1, size):
                    if (self.points[individual[k]]['priority'] >
                            self.points[individual[k - 1]]['priority']):
                        individual[k], individual[k - 1] = individual[k - 1], individual[k]
                        break

        return individual,

    def _evaluate_individual(self, individual: List[int]) -> Tuple[float, float]:
        if len(set(individual)) != len(self.points):
            return (-float('inf'), float('inf'))

        total_time = 0
        total_priority = 0
        priority_order_penalty = 0

        for k in range(len(individual)):
            i = individual[k]
            total_priority += self.points[i]['priority']

            if k > 0:
                prev_priority = self.points[individual[k - 1]]['priority']
                curr_priority = self.points[i]['priority']
                if curr_priority > prev_priority:
                    priority_order_penalty += (curr_priority - prev_priority) * 10

        for k in range(len(individual) - 1):
            i = individual[k]
            j = individual[k + 1]
            total_time += self.get_time(i, j)

        if len(individual) > 1:
            i = individual[-1]
            j = individual[0]
            total_time += self.get_time(i, j)

        if total_time > self.max_time:
            return (-float('inf'), float('inf'))

        return (total_priority, priority_order_penalty)

    def solve(self, population_size=300, generations=500):
        try:
            pop = self.toolbox.population(n=population_size)
            hof = tools.HallOfFame(10)

            stats = tools.Statistics(lambda ind: ind.fitness.values[0])
            stats.register("avg", np.mean)
            stats.register("min", np.min)
            stats.register("max", np.max)

            for gen in range(generations):
                # Динамические вероятности
                cx_prob = max(0.7 - 0.5 * gen / generations, 0.4)
                mut_prob = min(0.1 + 0.5 * gen / generations, 0.6)

                offspring = self.toolbox.select(pop, len(pop))
                offspring = [self.toolbox.clone(ind) for ind in offspring]

                # Кроссовер
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < cx_prob:
                        try:
                            child1[:], child2[:] = self.toolbox.mate(child1, child2)
                            del child1.fitness.values
                            del child2.fitness.values
                        except:
                            continue

                # Мутация
                for mutant in offspring:
                    if random.random() < mut_prob:
                        try:
                            self.toolbox.mutate(mutant)
                            del mutant.fitness.values
                        except:
                            continue

                # Оценка новых особей
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(self.toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # Отбор
                pop = tools.selBest(pop + offspring, population_size)
                hof.update(pop)

            best = hof[0]
            return {
                'route': best,
                'evaluation': self._evaluate_route(best)
            }
        except Exception as e:
            print(f"Ошибка в генетическом алгоритме: {str(e)}")
            return None

    def _evaluate_route(self, route: List[int]) -> Dict:
        total_priority = sum(self.points[i]['priority'] for i in route)
        total_time = 0
        total_distance = 0
        priority_order_penalty = 0

        for k in range(len(route)):
            if k > 0:
                prev_priority = self.points[route[k - 1]]['priority']
                curr_priority = self.points[route[k]]['priority']
                if curr_priority > prev_priority:
                    priority_order_penalty += (curr_priority - prev_priority) * 10

        for k in range(len(route) - 1):
            i = route[k]
            j = route[k + 1]
            total_time += self.get_time(i, j)
            total_distance += self.get_distance(i, j)

        if len(route) > 1:
            i = route[-1]
            j = route[0]
            total_time += self.get_time(i, j)
            total_distance += self.get_distance(i, j)

        return {
            'priority': total_priority,
            'time': total_time,
            'distance': total_distance,
            'priority_order_penalty': priority_order_penalty,
            'is_valid': total_time <= self.max_time and
                        len(set(route)) == len(self.points)
        }


def visualize_route(points: List[Dict], route: List[int], profile: str = 'car'):
    if not route:
        print("No route to visualize")
        return

    osrm = OSRMClient()
    route_points = [points[i] for i in route]
    coords = [(p['lat'], p['lon']) for p in route_points]

    try:
        osrm.get_route_info(coords, profile)

        m = folium.Map(location=coords[0], zoom_start=5)
        folium.PolyLine(
            locations=coords,
            color='blue',
            weight=3,
            opacity=0.7
        ).add_to(m)

        for i, point in enumerate(route_points):
            folium.Marker(
                location=(point['lat'], point['lon']),
                popup=f"{point['name']} (Приоритет: {point['priority']})",
                icon=folium.Icon(color='green' if i == 0 else 'red')
            ).add_to(m)

        return m
    except Exception as e:
        print(f"Error visualizing route: {str(e)}")
        return None


if __name__ == "__main__":
    cities = [
        {'name': 'Париж', 'lat': 48.8566, 'lon': 2.3522, 'priority': 5},
        {'name': 'Берлин', 'lat': 52.5200, 'lon': 13.4050, 'priority': 4},
        {'name': 'Рим', 'lat': 41.9028, 'lon': 12.4964, 'priority': 3},
        {'name': 'Мадрид', 'lat': 40.4168, 'lon': -3.7038, 'priority': 2},
        {'name': 'Амстердам', 'lat': 52.3676, 'lon': 4.9041, 'priority': 1},
    ]

    max_time = 9000

    print("Идеальный порядок по приоритетам:")
    for i, city in enumerate(sorted(cities, key=lambda x: -x['priority'])):
        print(f"{i + 1}. {city['name']} (приоритет: {city['priority']})")

    optimizer = GeneticOptimizer(cities, max_time, profile='car')
    solution = optimizer.solve(population_size=300, generations=500)

    if solution and solution['evaluation']['is_valid']:
        print("\nНайден оптимальный маршрут:")
        for i, idx in enumerate(solution['route']):
            print(f"{i + 1}. {cities[idx]['name']} (приоритет: {cities[idx]['priority']})")

        print(f"\nСуммарный приоритет: {solution['evaluation']['priority']}")
        print(f"Штраф за нарушение порядка: {solution['evaluation']['priority_order_penalty']}")
        print(f"Общее время: {solution['evaluation']['time']:.1f} мин")
        print(f"Общее расстояние: {solution['evaluation']['distance']:.1f} км")

        map_obj = visualize_route(cities, solution['route'])
        if map_obj:
            map_obj.save('optimized_route.html')
            print("\nКарта маршрута сохранена в optimized_route.html")
    else:
        print("Не удалось найти маршрут, удовлетворяющий ограничениям")