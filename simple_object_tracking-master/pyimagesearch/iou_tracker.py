# import the necessary packages
from scipy.spatial import distance as dist
from kalmanFilter import KalmanFilter
from collections import OrderedDict
import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment

np.set_printoptions(linewidth=220)

class Tracks(object):
    def __init__(self, detection, trackId):
        super(Tracks, self).__init__()
        self.KF = KalmanFilter()
        self.KF.predict()
        self.KF.correct(np.matrix(detection).reshape(2, 1))
        self.trace = deque(maxlen=50)
        self.prediction = detection.reshape(1, 2)
        self.trackId = trackId
        self.skipped_frames = 0

    def predict(self, detection):
        self.prediction = np.array(self.KF.predict()).reshape(1, 2)
        self.KF.correct(np.matrix(detection).reshape(2, 1))



class Tracker():
    def __init__(self, dist_threshold, max_frame_skipped, max_trace_length, iou_threshold):
        super(Tracker, self).__init__()
        self.dist_threshold = dist_threshold
        self.max_frame_skipped = max_frame_skipped
        self.trace = deque(maxlen=max_trace_length)
        self.max_trace_length = max_trace_length
        self.iou_threshold = iou_threshold
        self.trackId = 0
        self.tracks = []
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.objects_trace = OrderedDict()
        self.disappeared = OrderedDict()
        self.mixed_up = OrderedDict()
        self.max_leave_out = 2

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = max_frame_skipped

    def get_iou_score(self, box1: np.ndarray, box2: np.ndarray):
        """
        calculate intersection over union cover percent
        :param box1: box1 with shape (N,4) or (N,2,2) or (2,2) or (4,). first shape is preferred
        :param box2: box2 with shape (N,4) or (N,2,2) or (2,2) or (4,). first shape is preferred
        :return: IoU ratio if intersect, else 0
        """
        # first unify all boxes to shape (N,4)
        if box1.shape[-1] == 2 or len(box1.shape) == 1:
            box1 = box1.reshape(1, 4) if len(box1.shape) <= 2 else box1.reshape(box1.shape[0], 4)
        if box2.shape[-1] == 2 or len(box2.shape) == 1:
            box2 = box2.reshape(1, 4) if len(box2.shape) <= 2 else box2.reshape(box2.shape[0], 4)
        point_num = max(box1.shape[0], box2.shape[0])
        b1p1, b1p2, b2p1, b2p2 = box1[:, :2], box1[:, 2:], box2[:, :2], box2[:, 2:]

        # mask that eliminates non-intersecting matrices
        base_mat = np.ones(shape=(point_num,))
        base_mat *= np.all(np.greater(b1p2 - b2p1, 0), axis=1)
        base_mat *= np.all(np.greater(b2p2 - b1p1, 0), axis=1)

        # I area
        intersect_area = np.prod(np.minimum(b2p2, b1p2) - np.maximum(b1p1, b2p1), axis=1)
        # U area
        union_area = np.prod(b1p2 - b1p1, axis=1) + np.prod(b2p2 - b2p1, axis=1) - intersect_area
        # IoU
        if union_area.all():
            intersect_ratio = intersect_area / union_area
        else:
            intersect_ratio = 0

        return base_mat * intersect_ratio

    def register(self, coordinates):
        # coordinates in the format [xmin,ymin,xmax,ymax]
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = coordinates
        self.objects_trace[self.nextObjectID] = [self.get_centroid(coordinates)]
        self.disappeared[self.nextObjectID] = 0
        self.mixed_up[self.nextObjectID] = 0
        self.nextObjectID += 1


    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.objects_trace[objectID]
        del self.disappeared[objectID]
        del self.mixed_up[objectID]

    def get_centroid(self, coordinates):
        return (coordinates[0] + (coordinates[2] - coordinates[0]) // 2,
                coordinates[1] + (coordinates[3] - coordinates[1]) // 2)

    def get_min_distance_order(self, D):
        D_order = []
        for i in range(D.shape[1]):
            sorted_col = np.argsort(D[:, i])
            for j in range(D.shape[0]):
                if sorted_col[j] not in D_order:
                    D_order.append(sorted_col[j])
                    break
        return D_order

    def get_max_iou_order(self, iou_scores):
        I = np.absolute(np.array(iou_scores))
        I_order = []
        for i in range(I.shape[1]):
            sorted_col = np.argsort(-I[:, i])
            for j in range(I.shape[0]):
                if sorted_col[j] not in I_order:
                    I_order.append(sorted_col[j])
                    break
        return I_order

    def update(self, rects):
        detections = []
        inputCoordinates = np.array(rects)
        if len(self.tracks) == 0:
            for i in range(np.array(rects).shape[0]):
                centroid = self.get_centroid(rects[i])
                track = Tracks(np.array(list(centroid)), self.trackId)
                detections.append(list(centroid))
                self.register(inputCoordinates[i])
                self.trackId += 1
                self.tracks.append(track)

        else:
            objectIDs = list(self.objects.keys())
            objectCoordinates = list(self.objects.values())
            D = dist.cdist(np.array(objectCoordinates), inputCoordinates,'chebyshev')
            iou_scores = []
            for o in objectCoordinates:
                iou_scores.append(self.get_iou_score(np.array(o), np.array(inputCoordinates)))
            iou_scores = np.array(iou_scores)


            # order the distance matrix along the main diagonal with smallest values
            D_order = self.get_min_distance_order(np.array(D))



            # order the IOU matrix along the main diagonal with largest values
            I_order = self.get_max_iou_order(iou_scores)

            # merge both orderings and see which makes more sense
            order = []
            for i in range(len(D_order)):
                if D_order[i] == I_order[i]:
                    order.append(D_order[i])
                else:
                    if np.max(iou_scores[I_order,:][i]) == 0:
                        order.append(D_order[i])
                    else:
                        order.append(I_order[i])

            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value as at the *front* of the index
            # list
            #d_rows = D.min(axis=1).argsort()
            #rows = iou_scores.max(axis=1).argsort()
            # rows_end,cols_end = D.shape
            # rows = np.arange(0,rows_end)
            # cols = np.arange(0,cols_end)



            #TODO: prevent double ids,
            # prevent switch to an already existing object (play with direction of the trace)
            #
            #

            # print(rows)

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list

            #d_cols = D.argmin(axis=1)[rows]
            #cols = iou_scores.argmax(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index
            # tuples
            # for (row, col) in zip(rows, cols):
            for col, row in enumerate(order):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue

                # new_col_d = np.argmin(D[row])
                # new_col_iou = np.argmax(iou_scores[row])
                # if new_col_d == new_col_iou:
                #     new_col = new_col_d
                # else:
                #     new_col = col
                # print("best match: row {} and col {}".format(row,new_col))
                # if new_col in usedCols:
                #     new_col = col
                #
                # try:
                #     if new_col != col:
                #         if self.mixed_up[objectIDs[row]] >= self.max_leave_out:
                #             pass
                #         else:
                #             self.mixed_up[objectIDs[row]] += 1
                #             # usedRows.add(row)
                #             # usedCols.add(col)
                #             continue
                #
                # except:
                #     print(self.mixed_up[objectIDs[row]])
                #
                # try:
                #     inverted_index_d = D[new_col, row]
                # except:
                #     inverted_index_d = 999


                # if new_col == col:
                    # frame and frame+1 have the best match so they stay the same
                if iou_scores[row, col] >= self.iou_threshold or D[row, col] <= self.dist_threshold:
                    objectID = objectIDs[row]
                    self.objects[objectID] = inputCoordinates[col]
                    self.objects_trace[objectID].append(self.get_centroid(inputCoordinates[col]))
                    self.disappeared[objectID] = 0
                # elif D[row, new_col] < inverted_index_d and new_col != col:
                #     # here we give the bee a new id
                #     if new_col in rows:
                #         if iou_scores[row, new_col] >= self.iou_threshold or D[row, new_col] <= self.dist_threshold:
                #             print(row,new_col)
                #             print("Switching {} with {}".format(objectIDs[row],objectIDs[new_col]))
                #             objectID = objectIDs[row]
                #             self.objects[objectID] = inputCoordinates[new_col]
                #             self.objects_trace[objectID].append(self.get_centroid(inputCoordinates[new_col]))
                #             self.disappeared[objectID] = 0
                #
                #             objectID = objectIDs[new_col]
                #             self.objects[objectID] = inputCoordinates[row]
                #             self.objects_trace[objectID].append(self.get_centroid(inputCoordinates[row]))
                #             self.disappeared[objectID] = 0





                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                # switches between bees that are very close is okay
                # if iou_scores[row, col] >= self.iou_threshold or D[row, col] <= self.dist_threshold:
                #     objectID = objectIDs[row]
                #     self.objects[objectID] = inputCoordinates[col]
                #     self.objects_trace[objectID].append(self.get_centroid(inputCoordinates[col]))
                #     self.disappeared[objectID] = 0

                # elif D[row, col] >= self.dist_threshold:


                    # indicate that we have examined each of the row and
                    # column indexes, respectively
                    usedRows.add(row)
                    usedCols.add(col)
                else:
                    pass
                # print("iou: {}".format(iou_scores[row, col]))
                # print("D: {}".format(D[row, col]))
                # print(self.nextObjectID)

            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, iou_scores.shape[0])).difference(usedRows)
            unusedCols = set(range(0, iou_scores.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if iou_scores.shape[0] >= iou_scores.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCoordinates[col])

            # duplicates = []
            # for i in rows:
            #     for j in rows:
            #         if i!=j and i not in duplicates:
            #             if (D[i] == D[j]).all() and (iou_scores[i] == iou_scores[j]).all():
            #                 duplicates.append(j)
            #
            # if len(duplicates) > 0:
            #     print(duplicates)
            #     for i in duplicates:
            #         self.deregister(objectIDs[i])


        # detections = []
        # for rect in rects:
        #     detections.append(np.array(list(self.get_centroid(rect))))


        # N = len(self.tracks)
        # cost = []
        # for i in range(N):
        #     diff = np.linalg.norm(self.tracks[i].prediction - np.array(detections).reshape(-1, 2), axis=1)
        #     cost.append(diff)
        #
        # cost = np.array(cost) * 0.1
        # print(cost)
        # row, col = linear_sum_assignment(cost)
        # print(row,col)
        # assignment = [-1] * N
        # for i in range(len(row)):
        #     assignment[row[i]] = col[i]
        # print(assignment)
        #
        # un_assigned_tracks = []
        #
        # for i in range(len(assignment)):
        #     if assignment[i] != -1:
        #         print(cost[i][assignment[i]])
        #         print(self.dist_threshold)
        #         if (cost[i][assignment[i]] > self.dist_threshold):
        #             assignment[i] = -1
        #             un_assigned_tracks.append(i)
        #         else:
        #             self.tracks[i].skipped_frames += 1
        #     else:
        #         self.tracks[i].skipped_frames += 1
        #
        #
        # print(un_assigned_tracks)
        #
        # # check if amount of skipped frames is in the allowed numbers
        # del_tracks = []
        # for i in range(len(self.tracks)):
        #     if self.tracks[i].skipped_frames > self.max_frame_skipped:
        #         del_tracks.append(i)
        #
        #
        #
        # # deregister missing tracks if max frame skipped is reached
        # if len(del_tracks) > 0:
        #     for i in range(len(del_tracks)):
        #         print("deleting")
        #         del self.tracks[i]
        #         del assignment[i]
        #
        # for i in range(len(detections)):
        #     if i not in assignment:
        #         track = Tracks(detections[i], self.trackId)
        #         self.trackId += 1
        #         self.tracks.append(track)
        #
        # for i in range(len(assignment)):
        #     if assignment[i] != -1:
        #         self.tracks[i].skipped_frames = 0
        #         self.tracks[i].predict(detections[assignment[i]])
        #         self.tracks[i].trace.append(self.tracks[i].prediction)

        # return the set of trackable objects

        return self.objects, self.objects_trace
