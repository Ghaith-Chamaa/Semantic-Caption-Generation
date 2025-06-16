db.region_graphss.aggregate([{
        $sample: { size: 5 }
    },
    {
        $unwind: "$regions"
    },

    {
        $project: {
            regions: 1,
            _id: 0,
            image_id: 1

        }
    },
    {

        $lookup: {
            from: "Results",
            let: {
                localrelationship: { $ifNull: ["$regions.relationships.relationship_id", []] },
                localImage: "$image_id",
            },

            pipeline: [
                { $match: { $expr: { $eq: ["$image_id", "$$localImage"] } } },
                { $unwind: "$relationships" },
                {
                    $match: {
                        $expr: {
                            $in: [
                                "$relationships.relationships.relationship_id",
                                "$$localrelationship"
                            ]
                        }
                    }
                },


            ],
            as: "regions.relationships"

        }
    },
    {
        $lookup: {
            from: "newImages",
            let: {
                localImage: "$image_id",
                localObject: "$regions.objects.object_id"

            },
            pipeline: [
                { $match: { $expr: { $eq: ["$image_id", "$$localImage"] } } },
                { $unwind: "$objects" },
                { $unwind: "$objects.objects.object_id" },
                {
                    $match: {
                        $expr: { $eq: [{ $size: "$regions.relationships" }, 0] },
                        $expr: {

                            $in: ["$objects.objects.object_id.attributes.object_id", "$$localObject"]

                        }
                    }
                }





            ],
            as: "regions.objects"
        }
    },
    { $out: { db: "visual_genome", coll: "tors" } }
], { "allowDiskUse": true })