#/usr/bin/python2.7
#*-coding:utf-8-*-
from structure import DependencyTree
from execute import Execute
from time import clock as now
import os
import logging
def main():
    print 'ok.'
    lines = ('[(S (NP (DT Those) (NN space)) (VP (VBZ walks) (SBAR (S (VP (VBP are) (S (VP (TO to) (VP (VB be) (VP (VBN used) (PP (IN for) (S (VP (VBG preparing) (NP (NP (DT the) (NNP ISS)) (PP (IN for) (NP (NP (DT the) (VBN planned) (NN docking)) (NP (JJ next) (NN year)) (PP (IN of) (NP (DT the) (JJ new) (NNP European) (NNP ATV) (NN space) (NN cargo) (NN vessel))))))))))))))))) (. .))]','[det(space-2, Those-1), nsubj(walks-3, space-2), root(ROOT-0, walks-3), ccomp(walks-3, are-4), aux(used-7, to-5), auxpass(used-7, be-6), xcomp(are-4, used-7), prepc_for(used-7,preparing-9), det(ISS-11, the-10), dobj(preparing-9, ISS-11), det(docking-15, the-13), amod(docking-15, planned-14), prep_for(ISS-11, docking-15), amod(year-17, next-16), dep(docking-15, year-17), det(vessel-25, the-19), amod(vessel-25, new-20), nn(vessel-25, European-21), nn(vessel-25, ATV-22), nn(vessel-25, space-23), nn(vessel-25, cargo-24), prep_of(docking-15, vessel-25)]')

    D = DependencyTree()
    w = D.read_sent(lines[0])   
    print ' '.join([x[0]+'/'+x[1] for x in w])

    y = D.read_dependency(lines[1])
    for d in y:
        print d
    pass

def depCheck():
    path = '/home/gujt/work/Stanford_Sparser/work/Aquant2-ctree/0/'
    D = DependencyTree()
    max = 0
    for dir in os.walk(path):
        for file in dir[2]:
            print 'read.',file
            f = open(path+file)
            line = f.readline()
            while line:
                line = f.readline()
                try:
                    y = D.read_dependency(line)
                except:
                    print line
                    line = f.readline()
                    continue
                p = [h[0] for h in y]
                t = {}
                for pi in p:
                    if pi not in t:
                        t[pi] = 1
                    else:
                        t[pi] +=1
                t = sorted(t.iteritems(), key =lambda b:b[1], reverse=True)
                #print t
                try:
                    if t[0][1] > max: 
                        max = t[0][1]
                        print max
                except:
                    print t
                    
                #for yi in y:
                #   if yi[4] not in depDict:
                #       depDict[yi[4]] = 1
                #   else:
                #       depDict[yi[4]] += 1
                line = f.readline()
                            
            f.close()
        break   
    
    print max

    #fd = open('../depcheck2', 'w')
    #depDict = sorted(depDict.iteritems(), key=lambda x:x[0])
    #for dep in depDict:
    #   fd.write(dep[0]+' '+str(dep[1])+'\n')
    #fd.close()

def main3():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    path = '/home/gujt/work/data/Aquant2-ctree/0/'
    exe = Execute()
    #exe.build_model()
    exe.load_model()
    #exit()  
    count = 0
    for dir in os.walk(path):
        for file in dir[2]:
            print 'read.',file
            f = open(path+file)
            line1 = f.readline()
            start = now()
            while line1:
                if count > 10:
                    exit()
                count += 1
                line2 = f.readline()
                lines = [line1, line2]
                try:
                    exe.train_sentence(lines)
                except:
                    '[BUG]',line2
                finally:
                    line1 = f.readline()
                    print '[',count,']', (now()-start),'s'
                    start = now()
                            
            f.close()
    
    print 'train ok.'
    exe.save_weights('../data/weights.bin') 


if __name__ == "__main__":
    #depCheck()
    main3()
