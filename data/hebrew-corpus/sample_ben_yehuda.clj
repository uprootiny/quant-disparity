#!/usr/bin/env bb
;; Sample texts from Project Ben-Yehuda (public domain Hebrew literature)
;; Usage: bb sample_ben_yehuda.clj [n-samples]
;; Default: 10 samples, with 2-second delays between requests

(require '[babashka.http-client :as http]
         '[clojure.string :as str]
         '[clojure.java.io :as io])

(def base-url "https://raw.githubusercontent.com/projectbenyehuda/public_domain_dump/master")
(def catalogue-url (str base-url "/pseudocatalogue.csv"))
(def output-dir "samples")
(def delay-ms 2000) ; Be gentle on the server

(defn fetch-url [url]
  (Thread/sleep delay-ms)
  (println "  Fetching:" (subs url (max 0 (- (count url) 60))))
  (-> (http/get url {:throw false})
      :body))

(defn parse-csv-line [line]
  ;; Simple CSV parsing (handles quoted fields)
  (let [fields (atom [])
        current (atom "")
        in-quotes (atom false)]
    (doseq [c line]
      (cond
        (and (= c \") (not @in-quotes))
        (reset! in-quotes true)

        (and (= c \") @in-quotes)
        (reset! in-quotes false)

        (and (= c \,) (not @in-quotes))
        (do (swap! fields conj @current)
            (reset! current ""))

        :else
        (swap! current str c)))
    (swap! fields conj @current)
    @fields))

(defn fetch-catalogue []
  (println "Fetching catalogue...")
  (let [csv (fetch-url catalogue-url)
        lines (str/split-lines csv)
        header (parse-csv-line (first lines))
        rows (map parse-csv-line (rest lines))]
    (map #(zipmap header %) rows)))

(defn select-samples [catalogue n]
  ;; Select samples that have valid paths
  ;; Paths are like /p23/m10 which map to txt_stripped/p23/m10.txt
  (let [valid (filter #(re-matches #"/p\d+/m\d+" (get % "path" "")) catalogue)
        shuffled (shuffle valid)]
    (take n shuffled)))

(defn download-text [{:strs [path title authors] :as entry}]
  (let [;; Path like /p23/m10 -> txt_stripped/p23/m10.txt
        txt-path (str "txt_stripped" path ".txt")
        url (str base-url "/" txt-path)
        content (fetch-url url)]
    (when (and content
               (not (str/blank? content))
               (not (str/starts-with? content "404")))
      {:title title
       :author authors
       :path txt-path
       :content content
       :chars (count content)})))

(defn ensure-dir [dir]
  (.mkdirs (io/file dir)))

(defn save-sample [{:keys [title author content]} idx]
  (ensure-dir output-dir)
  (let [title-str (or title "unknown")
        full-name (str idx "_" title-str)
        safe-name (str/replace full-name #"[^\w\u0590-\u05FF]+" "_")
        truncated (subs safe-name 0 (min 50 (count safe-name)))
        filename (str output-dir "/" truncated ".txt")]
    (spit filename content)
    (println "  Saved:" filename)))

(defn -main [& args]
  (let [n-samples (or (some-> args first parse-long) 10)]
    (println "=== Ben-Yehuda Sampler ===")
    (println "Sampling" n-samples "texts with" delay-ms "ms delay")
    (println)

    (let [catalogue (fetch-catalogue)
          _ (println "Found" (count catalogue) "entries in catalogue")
          samples (select-samples catalogue n-samples)
          _ (println "Selected" (count samples) "random samples")
          _ (println)
          downloaded (keep download-text samples)
          total-chars (reduce + (map :chars downloaded))]

      (println)
      (println "=== Summary ===")
      (println "Downloaded:" (count downloaded) "texts")
      (println "Total chars:" total-chars)
      (println "Avg chars/text:" (if (pos? (count downloaded))
                                   (quot total-chars (count downloaded))
                                   0))

      ;; Save samples
      (doseq [[idx sample] (map-indexed vector downloaded)]
        (save-sample sample idx))

      ;; Save manifest
      (ensure-dir output-dir)
      (spit (str output-dir "/manifest.edn")
            (pr-str {:source "Project Ben-Yehuda"
                     :url "https://github.com/projectbenyehuda/public_domain_dump"
                     :license "Public Domain"
                     :samples (mapv #(dissoc % :content) downloaded)
                     :total-chars total-chars
                     :timestamp (str (java.time.Instant/now))}))

      (println)
      (println "Manifest saved to" (str output-dir "/manifest.edn")))))

(when (= *file* (System/getProperty "babashka.file"))
  (apply -main *command-line-args*))
