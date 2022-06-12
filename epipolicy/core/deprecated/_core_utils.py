import numba as nb
import numpy as np
from numba.typed import Dict, List
from numba.types import string

def get_tag_count(objs):
    tags = set()
    for obj in objs:
        if "tags" in obj:
            for tag in obj["tags"]:
                tags.add(tag)
    return len(tags)

def get_locale_count(locales):
    parents = set()
    idToIndex = {}
    for i, locale in enumerate(locales):
        idToIndex[locale["id"]] = i
        parents.add(locale["parent_id"])
    truncatedLocales = []
    for locale in locales:
        if locale["id"] not in parents:
            truncatedLocales.append(locale)
    return idToIndex, truncatedLocales

def setLocales(locales, static):
    idToIndex = {}
    parents = set()
    truncatedLocales = []
    for i, locale in enumerate(locales):
        idToIndex[locale["id"]] = i
        static.localeHierarchy[normalize(locale["name"])] = Dict.empty(nbInt, nbBool)
        parents.add(locale["parent_id"])
    for i, locale in enumerate(locales):
        locId = locale["id"]
        if locId not in parents:
            # leaf locale
            truncatedLocales.append(locale)
            static.addProp("locale", locId)
            curLocale = locales[i]
            while True:
                static.localeHierarchy[normalize(curLocale["name"])][static.getPropIndex("locale", locId)] = True
                parentId = curLocale["parent_id"]
                if len(parentId) > 0:
                    curLocale = locales[idToIndex[parentId]]
                else:
                    break
    static.localeHierarchy[""] = Dict.empty(nbInt, nbBool)
    for i in range(static.nLocales):
        static.localeHierarchy[""][i] = True
    return truncatedLocales

def setGroups(groups, static):
    if len(groups) == 0:
        static.addProp("group", "All")
    else:
        for group in groups:
            static.addProp("group", group["name"])

def setCompartments(compartments, static):
    # Populate alive vector, infectious vector, susceptible vector and initId
    for com in compartments:
        index = static.addProp("compartment", com["name"])
        for tag in com["tags"]:
            tagIndex = static.addProp("compartmentTag", tag)
            static.compartmentTags[index, tagIndex] = 1
            # Need a better solution here when there are multiple INI_TAG
            # if tag == INI_TAG:
            #     static.initId = index

def setFacilities(facilities, static):
    if len(facilities) == 0:
        static.addProp("facility", "All")
    else:
        for facility in facilities:
            static.addProp("facility", facility["name"])

def populate(locales, groups, static):
    # Populate initial state matrix of size cxlxg
    # if static.nGroups == 1:
    #     for i, locale in enumerate(locales):
    #         static.s[static.initId, i, 0] = float(locale["population"])
    # else:
    #     for i, group in enumerate(groups):
    #         for locale in group["locales"]:
    #             static.s[static.initId, static.getPropIndex("locale", locale["id"]), i] = locale["population"]
    #     for j, locale in enumerate(locales):
    #         for i in range(static.nGroups):
    #             static.s[static.initId, j, i] *= float(locale["population"])
    if static.nGroups == 1:
        for i, locale in enumerate(locales):
            static.sN[i, 0] = float(locale["population"])
    else:
        for i, group in enumerate(groups):
            for locale in group["locales"]:
                static.sN[static.getPropIndex("locale", locale["id"]), i] = float(locale["population"])
        for j, locale in enumerate(locales):
            for i in range(static.nGroups):
                static.sN[j, i] *= float(locale["population"])
    #static.sN = np.sum(static.s, axis=0)
    static.pN = static.sN/np.sum(static.sN, axis=1)

def setParameters(groups, parameters, customParameters, static, parser):
    for param in parameters:
        paramId = static.addProp("parameter", param["name"])
        value = float(param["default_value"])
        static.P.p[paramId] = np.ones((static.nLocales, static.nFacilities, static.nGroups), dtype=npFloat)*value

        for tag in param.get("tags", []):
            tagId = static.addProp("parameterTag", tag)
            static.parameterTags[paramId, tagId] = 1

    for param in customParameters:
        paramId = static.getPropIndex("parameter", param["param"])
        locales = parser.parseLRegex(param["locale"])
        #if param["group"] in static.groupToIndex: # Temporary fix to ignore "Others"
        groupId = static.getPropIndex("group", param["group"])
        for l in locales:
            static.P.p[paramId, l, :, groupId] = float(param["value"])

    #Customize parameters - temporary
    # isCustomizable = True
    # if static.nGroups != 3:
    #     isCustomizable = False
    # if isCustomizable:
    #     for i, g in enumerate(groups):
    #         if g["name"] != GROUPS[i]:
    #             isCustomizable = False
    # if isCustomizable:
    #     for param in PARAMETERS:
    #         if param not in static.parameterToIndex:
    #             isCustomizable = False
    # if isCustomizable:
    #     for i, param in enumerate(PARAMETERS):
    #         paramId = static.parameterToIndex[param]
    #         for j, group in enumerate(GROUPS):
    #             groupId = static.groupToIndex[group]
    #             for l in range(static.nLocales):
    #                 static.p[paramId, l, groupId] = PARAMETER_VALUES[i, j]
    #     print("Finished customizing parameters", flush=True)

def setModes(static):
    for mode in MODES:
        static.addProp("mode", mode)

def setBias(static):
    static.bias[static.getPropIndex("mode", "airport")] = 0
    static.bias[static.getPropIndex("mode", "border")] = 1

def setLocaleMobility(session, mode, static, parser):
    mobilities = session["{}".format(mode)]["data"]
    coos = static.coo[static.getPropIndex("mode", mode)]
    for mob in mobilities:
        groups = parser.parseGRegex(mob["group"]).data
        src = static.getPropIndex("locale", mob["src_locale_id"])
        dst = static.getPropIndex("locale", mob["dst_locale_id"])
        val = float(mob["value"])
        for g in groups:
            coos[g].set(src, dst, val)

    defaultWithin = WITHIN_RATIO_FACTILITY if static.nLocales > 1 else 1.0
    specifications = session["{}".format(mode)]["specifications"]
    if len(specifications) > 0:
        if 'impedance' in specifications[0]:
            defaultWithin = float(specifications[0]['impedance'])/100
    for i in range(static.nGroups):
        for l in range(static.nLocales):
            within = defaultWithin
            if coos[i].has(l, l):
                within = coos[i].get(l, l)
            coos[i].set(l, l, within)
            for dest, v in coos[i]._from[l].items():
                if dest != l:
                    coos[i].set(l, dest, v*(1-within))

def setLocaleMatrix(session, static, parser):
    coos = List.empty_list(cooMatrixType)
    for i, mode in enumerate(MODES):
        setLocaleMobility(session, mode, static, parser)
        coo = CooMatrix(static.shape).addSignature(static.coo[i])
        coos.append(coo)
        static.csr.append(getCsrMatrix(coo))

    sumCoo = CooMatrix(static.shape).addSignature(coos)
    static.sumCsr = getCsrMatrix(sumCoo)
    static.sumCsc = getCscMatrix(sumCoo)

    for i in range(len(MODES)):
        csrData = np.array([getCsrMatrix(static.coo[i][j].addSignature(coos[i:i+1])).data for j in range(static.nGroups)], dtype=npFloat)
        static.P.csr.append(Np2f(csrData))

def setFacilityMatrix(session, static, parser):
    # Populate the facility matrix of size lxfxg
    # if contactMatrix is not None:
    #     fgMatrix = np.sum(contactMatrix, axis=2)
    #     gMatrix = np.sum(fgMatrix, axis=0)
    #     fgMatrix /= gMatrix
    #     for i in range(static.nLocales):
    #         static.f[i] = fgMatrix.copy()
    if len(session["facilities_timespent"]) > 0:
        for timespent in session["facilities_timespent"]:
            locales = parser.parseLRegex(timespent["locales"])
            for l in locales:
                static.P.f[l] = np.array(timespent["matrix"])[:static.nFacilities,:static.nGroups]

def setContactMatrix(session, static, parser):
    # Populate the contact matrix of size lxfxgxg
    # if contactMatrix is not None:
    #     fgMatrix = np.sum(contactMatrix, axis=2)
    #     fggMatrix = np.zeros_like(contactMatrix)
    #     for i in range(static.nFacilities):
    #         for j in range(static.nGroups):
    #             fggMatrix[i,j] = contactMatrix[i,j] / fgMatrix[i,j]
    #     for i in range(static.nLocales):
    #         static.ft[i] = fggMatrix.copy()
    if len(session["facilities_interactions"]) > 0:
        for interaction in session["facilities_interactions"]:
            locales = parser.parseLRegex(interaction["locales"])
            for l in locales:
                static.P.ft[l] = np.array(interaction["facilities"])[:static.nFacilities, :static.nGroups, :static.nGroups]

# def setFacilityMobilities(session, static):
    # contactMatrix = np.zeros((static.nFacilities, static.nGroups, static.nGroups), dtype=npFloat)
    # groups = session["groups"]
    # facilities = session["facilities"]
    # if static.nFacilities != 3 or static.nGroups != 3:
    #     contactMatrix = None
    # if contactMatrix is not None:
    #     for i, fac in enumerate(facilities):
    #         if fac["name"] != FACILITIES[i]:
    #             contactMatrix = None
    # if contactMatrix is not None:
    #     for i in range(static.nGroups):
    #         if len(groups[i]["ds_groups"]) == 0:
    #             contactMatrix = None
    # if contactMatrix is not None:
    #     currentFolder = Path(__file__).parent
    #     path = (currentFolder / "contact_matrix.npy").resolve()
    #     f = open(path, 'rb')
    #     rawMatrix = np.load(f)
    #     f.close()
    #     tuplesToIndex = {}
    #     for i in range(static.nGroups - 1):
    #         for rangeInfo in groups[i]["ds_groups"]:
    #             genderAge = (0 if rangeInfo['gender'][0] == 'f' else 1, rangeInfo['age'][0]//AGE_GROUP)
    #             tuplesToIndex[genderAge] = i
    #     nGenders = rawMatrix.shape[1]
    #     nRanges = rawMatrix.shape[2]
    #     for s in range(nGenders):
    #         for g in range(nRanges):
    #             if (s, g) not in tuplesToIndex:
    #                 tuplesToIndex[(s, g)] = static.nGroups - 1
    #     for f in range(static.nFacilities):
    #         for s1 in range(nGenders):
    #             for g1 in range(nRanges):
    #                 for s2 in range(nGenders):
    #                     for g2 in range(nRanges):
    #                         index1 = tuplesToIndex[(s1, g1)]
    #                         index2 = tuplesToIndex[(s2, g2)]
    #                         contactMatrix[f, index1, index2] += rawMatrix[f, s1, g1, s2, g2]
    #     print("Finished customizing contact pattern")

# def setJacSparsityMatrix(static):
#     dim = static.nCompartments*static.nLocales*static.nGroups+1
#     coo = CooMatrix((dim, dim))
#     for i in range(static.nLocales):
#         influencers = []
#         for ind in range(static.sumCsc.indptr[i], static.sumCsc.indptr[i+1]):
#             for c in range(static.nCompartments):
#                 for g in range(static.nGroups):
#                     influencers.append(c*static.nGroups*static.nLocales+static.sumCsc.indices[ind]*static.nGroups+g)
#         for r in influencers:
#             for c in influencers:
#                 coo.set(r, c, 1)
#     for j in range(dim-1):
#         coo.set(dim-1, j, 1)
#     static.jacSparsity = getCsrMatrix(coo)

def checkInfectionEdge(frac, static):
    numer = frac.numer.muls
    param, sus, inf = None, None, None
    numerParams = []
    denomParams = []
    isInfectionEdge = True
    for s in numer:
        if static.hasPropName("parameter", s):
            index = static.toIndex["parameter"][s]
            if static.hasPropName("parameterTag", TRANS_TAG):
                if static.parameterTags[index, static.getPropIndex("parameterTag", TRANS_TAG)] > 0:
                    if param is None:
                        param = index
                    else:
                        isInfectionEdge = False
                else:
                    numerParams.append(index)
        elif static.hasPropName("compartment", s):
            index = static.toIndex["compartment"][s]
            if static.hasPropName("compartmentTag", INF_TAG):
                if static.compartmentTags[index, static.getPropIndex("compartmentTag", INF_TAG)] > 0:
                    if inf is None:
                        inf = index
                    else:
                        isInfectionEdge = False
            if static.hasPropName("compartmentTag", SUS_TAG):
                if static.compartmentTags[index, static.getPropIndex("compartmentTag", SUS_TAG)] > 0:
                    if sus is None:
                        sus = index
                    else:
                        isInfectionEdge = False
    if len(frac.denom) != 1:
        isInfectionEdge = False
    else:
        denom = frac.denom[0].muls
        hasAliveCompartment = False
        for s in denom:
            if s == ALIVE_COMPARTMENT:
                if hasAliveCompartment == False:
                    hasAliveCompartment = True
                else:
                    isInfectionEdge = False
            elif static.hasPropName("parameter", s):
                denomParams.append(static.toIndex["parameter"][s])
        if hasAliveCompartment == False:
            isInfectionEdge = False
        # if len(denom) != 1 or denom[0] != ALIVE_COMPARTMENT:
        #     isInfectionEdge = False
    if param == None or sus == None or inf == None:
        isInfectionEdge = False
    return isInfectionEdge, param, sus, inf, np.array(numerParams, dtype=npInt), np.array(denomParams, dtype=npInt)

def setNewInfEdges(static, edge):
    inCom = []
    outCom = []
    for com, coef in edge.coms.items():
        if coef > 0:
            inCom.append(com)
        else:
            outCom.append(com)
    if static.hasPropName("compartmentTag", SUS_TAG):
        for c1 in outCom:
            if static.compartmentTags[c1, static.getPropIndex("compartmentTag", SUS_TAG)] > 0:
                for c2 in inCom:
                    static.hashedNewInfEdges[c1*static.nCompartments+c2] = True

def setEdges(compartments, static):
    for i, com in enumerate(compartments):
        temp = com["equation"].replace('\n','')
        #print(temp)
        eqParser = EqParser(temp)
        fracs = eqParser.getFractions()
        for frac in fracs:
            infCheck = checkInfectionEdge(frac, static)
            key = frac.signature
            #print(key, infCheck)
            if not infCheck[0]:
                if key not in static.edges:
                    static.edges[key] = Edge(frac, infCheck[1] is not None)
                #print("NORMAL", key, frac.getCoef())
                static.edges[key].addComs(i, frac.getCoef())
            else:
                if key not in static.infEdges:
                    static.infEdges[key] = InfEdge(frac, *infCheck[1:])
                #print("INF", key, frac.getCoef())
                edge = static.infEdges[key]
                #print(edge.numerParamIds, edge.denomParamIds, edge.paramId, edge.susId, edge.infId)
                static.infEdges[key].addComs(i, frac.getCoef())
    for edge in static.edges.values():
        if edge.hasTransmissionRate:
            setNewInfEdges(static, edge)
    for edge in static.infEdges.values():
        setNewInfEdges(static, edge)

def seed(initializers, truncatedLocales, groups, compartments, static, parser):
    # Temporarily makeup value since UI has not done yet
    # CHOICE = 0
    # if CHOICE == 0:
    N = np.sum(static.sN, axis=1)
    for initializer in initializers:
        locIds = parser.parseLRegex(initializer['locale_regex'])
        totalPop = 0
        for l in locIds:
            totalPop += N[l]
        val = float(initializer['value'])
        groupIds = parser.parseGRegex(initializer['group']).data
        comIds = parser.parseCRegex(initializer['compartment']).data
        nPartitions = len(groupIds) * len(comIds)
        if nPartitions > 0:
            for l in locIds:
                for g in groupIds:
                    for c in comIds:
                        #static.s[static.initId, l, g] -= val * N[l] / totalPop / nPartitions
                        static.s[c, l, g] += val * N[l] / totalPop / nPartitions
    SN = getSumTagMatrix(static, static.s, SUS_TAG)
    for l in range(static.nLocales):
        for g in range(static.nGroups):
            total = np.sum(static.s[:, l, g])
            if SN[l, g] > 0:
                static.s[:, l, g] *= static.sN[l, g] / total
            else:
                if static.hasPropName("compartmentTag", SUS_TAG):
                    susIndex = static.getPropIndex("compartmentTag", SUS_TAG)
                    nSusCompartments = 0
                    for c in range(static.nCompartments):
                        if static.compartmentTags[c, susIndex] > 0:
                            nSusCompartments += 1
                    left = max(static.sN[l, g] - total, 0)
                    for c in range(static.nCompartments):
                        if static.compartmentTags[c, susIndex] > 0:
                            static.s[c, l, g] += left/nSusCompartments

    # elif CHOICE == 1:
    #     val = 15
    #     #val = 17000
    #     N = np.sum(getSumNotTagMatrix(static, static.s, DIE_TAG), axis=1)
    #     totalPop = np.sum(N)
    #     for id_, locale in enumerate(truncatedLocales):
    #         for i, com in enumerate(compartments):
    #             if static.ivec[i] > 0:
    #                 static.s[static.initId, id_] -= val * N[id_] / totalPop
    #                 static.s[i, id_] += val * N[id_] / totalPop
    #                 break
    # elif CHOICE == 2:
    #     #f = open('simulator/test/core/us_filtered.csv', 'r')
    #     f = open('test/core/us_filtered.csv', 'r')
    #     dict = {}
    #     for l in f:
    #         name, death, active = l.split(',')
    #         name = name.replace(' ', '').lower()
    #         death = float(death)
    #         active = float(active)
    #         dict[name] = (death, active)
    #     f.close()
    #     N = np.sum(getSumNotTagMatrix(static, static.s, DIE_TAG), axis=1)
    #     TOTAL_VACCINE = 800000
    #     FULLY_VACCINATED = 27300000
    #     totalPop = np.sum(N)
    #     partitionId = [2, 3, 4, 5, 6]
    #     ASYM = 0.33
    #     PRESYM = 0.62
    #     MILD = 0.81
    #     SEV = 0.14
    #     CRI = 0.05
    #     left = 1 - ASYM - PRESYM
    #     partition = [PRESYM, ASYM, left*MILD, left*SEV, left*CRI]
    #     initializers = []
    #     death_rate = [0.005, 0.28, 0.715]
    #     res = ""
    #     for id_, locale in enumerate(truncatedLocales):
    #         name = locale['name'].split('.')[1].replace(' ', '').lower()
    #         ######
    #         sample = """
    #         {}_doses = {}
    #         apply("compartment:S compartment:V locale:{} group:Seniors", administration_rate*{}_doses*efficacy*seniors_adults_ratio)
    #         apply("compartment:S compartment:V locale:{} group:Adults", administration_rate*{}_doses*efficacy*(1-seniors_adults_ratio))"""
    #         sample2 = """
    #         {}_doses = {}
    #         apply("influence:{} flat:True compartment:S locale:{} group:Seniors".format(seniors_percentage*{}_doses), {}_doses*seniors_percentage*dose_price)
    #         apply("influence:{} flat:True compartment:S locale:{} group:Adults".format(adults_percentage*{}_doses), {}_doses*adults_percentage*dose_price)"""
    #         originName = locale['name']
    #         doses = int(TOTAL_VACCINE * N[id_] / totalPop)
    #         #res += sample.format(name, doses, originName, name, originName, name)
    #         res += sample2.format(name, doses, "{}", originName, name, name, "{}", originName, name, name)
    #         ######
    #         if name in dict:
    #             death, active = dict[name]
    #             for g in range(static.nGroups):
    #                 ratio = static.s[static.initId, id_, g] / N[id_]
    #                 static.s[static.initId, id_, g] -= death_rate[g] * death
    #                 static.s[7, id_, g] += death_rate[g] * death
    #                 #
    #                 initialize = {
    #                     "id": len(initializers)+1,
    #                     "locale_regex": locale['name'],
    #                     "group": groups[g]['name'],
    #                     "compartment": compartments[7]['name'],
    #                     "value": death_rate[g] * death
    #                 }
    #                 initializers.append(initialize)
    #                 #
    #                 static.s[static.initId, id_, g] -= ratio * active
    #                 for i, pid in enumerate(partitionId):
    #                     static.s[pid, id_, g] += ratio * partition[i] * active
    #                     #
    #                     initialize = {
    #                         "id": len(initializers)+1,
    #                         "locale_regex": locale['name'],
    #                         "group": groups[g]['name'],
    #                         "compartment": compartments[pid]['name'],
    #                         "value": ratio * partition[i] * active
    #                     }
    #                     initializers.append(initialize)
    #                     #
    #             bigRatio = N[id_] / totalPop
    #             for g in range(1, static.nGroups):
    #                 ratio = static.s[static.initId, id_, g] / (N[id_] - static.s[static.initId, id_, 0])
    #                 static.s[static.initId, id_, g] -= ratio * bigRatio * FULLY_VACCINATED
    #                 static.s[14, id_, g] += ratio * bigRatio * FULLY_VACCINATED
    #                 #
    #                 initialize = {
    #                     "id": len(initializers)+1,
    #                     "locale_regex": locale['name'],
    #                     "group": groups[g]['name'],
    #                     "compartment": compartments[14]['name'],
    #                     "value": ratio * bigRatio * FULLY_VACCINATED
    #                 }
    #                 initializers.append(initialize)
    #                 #
    #     #print(res)
    #     f = open('US_Seeder.json', 'w')
    #     json.dump(initializers, f)
    #     f.close()

def setInterventions(interventions, costs, static):
    for itv in interventions:
        itvId = static.addProp("intervention", itv["name"])
        intervention = Intervention(itvId, itv["name"], False)
        for cpId, cp in enumerate(itv["control_params"]):
            name = cp["name"]
            defaultValue = float(cp["default_value"])
            intervention.cps.append(ControlParameter(cpId, name, defaultValue))
        #intervention.setHashLimit()
        static.interventions.append(intervention)
    offset = len(static.interventions)
    for cost in costs:
        costId = static.addProp("intervention", cost["name"])
        costIntervention = Intervention(offset+costId, cost["name"], True)
        for cpId, cp in enumerate(cost["control_params"]):
            name = cp["name"]
            defaultValue = float(cp["default_value"])
            costIntervention.cps.append(ControlParameter(cpId, name, defaultValue))
        static.interventions.append(costIntervention)

def getAction(itvId, detail, static):
    cpv = np.zeros(len(static.interventions[itvId].cps), dtype=npFloat)
    for cpId, cp in enumerate(detail['control_params']):
        cpv[cpId] = float(cp['value'])
    return Action(itvId, cpv, detail['locales'])

def setSchedule(features, schedules, static):
    startDate = parseDate(features["start_date"])
    endDate = parseDate(features["end_date"])
    T = (endDate - startDate).days
    static.schedule = Schedule((0, T), static.generateEmptyActions())
    for schedule in schedules:
        itvId = static.getPropIndex("intervention", schedule['name'])
        for detail in schedule['detail']:
            startT = (parseDate(detail['start_date']) - startDate).days
            endT = (parseDate(detail['end_date']) - startDate).days
            #print(startT, endT)
            for t in np.arange(startT, endT+1):
                static.schedule.addAction(static, t, getAction(itvId, detail, static))
    for t in np.arange(0, T):
        for itvId, itv in enumerate(static.interventions):
            if itv.isCost:
                static.schedule.addAction(static, t, generateDefaultAction(static, itvId))
        # If action not present, assuming it is a Zero Action
        # for itvId, itv in enumerate(static.interventions):
        #     if not itv.isCost and itvId not in static.schedule.ticks[t]:
        #         static.schedule.addAction(t, generateZeroAction(static, itvId))

def makeStatic(session, static, parser):
    features = session['features']
    model = session["model"]
    locales = session["locales"]
    groups = session["groups"]
    facilities = session["facilities"]
    compartments = model["compartments"]
    parameters = model["parameters"]
    customParameters = session["groups_locales_parameters"]
    interventions = session["interventions"]
    schedules = session["schedules"]
    costs = session["costs"]
    initializers = session["initial_info"]["initializers"]

    truncatedLocales = setLocales(locales, static)
    setGroups(groups, static)
    setFacilities(facilities, static)
    setCompartments(compartments, static)
    populate(truncatedLocales, groups, static)
    setParameters(groups, parameters, customParameters, static, parser)
    setModes(static)
    setBias(static)

    setLocaleMatrix(session, static, parser)
    setFacilityMatrix(session, static, parser)
    setContactMatrix(session, static, parser)

    #setJacSparsityMatrix(static)

    setEdges(compartments, static)
    seed(initializers, truncatedLocales, groups, compartments, static, parser) # TODO
    setInterventions(interventions, costs, static)
    setSchedule(features, schedules, static)
